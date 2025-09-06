import { GoogleGenerativeAI, GenerativeModel } from "@google/generative-ai";
import type { Channel, Event, MessageResponse, StreamChat } from "stream-chat";

export class GeminiResponseHandler {
  private message_text = "";
  private last_update_time = 0;
  private is_done = false;
  private abortController?: AbortController;

  constructor(
    private readonly genAI: GoogleGenerativeAI,
    private readonly model: GenerativeModel,
    private readonly chatClient: StreamChat,
    private readonly channel: Channel,
    private readonly message: MessageResponse,
    private readonly onDispose: () => void
  ) {
    this.chatClient.on("ai_indicator.stop", this.handleStopGenerating);
  }

  run = async () => {
    const { cid, id: message_id } = this.message;
    this.abortController = new AbortController();

    try {
      // Start generating response
      await this.channel.sendEvent({
        type: "ai_indicator.update",
        ai_state: "AI_STATE_GENERATING",
        cid: cid,
        message_id: message_id,
      });

      // Generate content with streaming
      const result = await this.model.generateContentStream(
        this.message_text,
        {
          signal: this.abortController.signal,
        }
      );

      let fullResponse = "";
      
      for await (const chunk of result.stream) {
        if (this.is_done) break;
        
        const chunkText = chunk.text();
        if (chunkText) {
          fullResponse += chunkText;
          
          // Update message every second to avoid too many updates
          const now = Date.now();
          if (now - this.last_update_time > 1000) {
            await this.chatClient.partialUpdateMessage(message_id, {
              set: { text: fullResponse },
            });
            this.last_update_time = now;
          }
        }
      }

      // Final update with complete response
      await this.chatClient.partialUpdateMessage(message_id, {
        set: { text: fullResponse },
      });

      // Clear AI indicator
      await this.channel.sendEvent({
        type: "ai_indicator.clear",
        cid: cid,
        message_id: message_id,
      });

    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        console.log("Generation was cancelled");
        await this.channel.sendEvent({
          type: "ai_indicator.clear",
          cid: cid,
          message_id: message_id,
        });
      } else {
        console.error("Error during generation:", error);
        await this.handleError(error as Error);
      }
    } finally {
      await this.dispose();
    }
  };

  dispose = async () => {
    if (this.is_done) {
      return;
    }
    this.is_done = true;
    this.chatClient.off("ai_indicator.stop", this.handleStopGenerating);
    this.onDispose();
  };

  private handleStopGenerating = async (event: Event) => {
    if (this.is_done || event.message_id !== this.message.id) {
      return;
    }

    console.log("Stop generating for message", this.message.id);
    
    if (this.abortController) {
      this.abortController.abort();
    }

    await this.channel.sendEvent({
      type: "ai_indicator.clear",
      cid: this.message.cid,
      message_id: this.message.id,
    });
    await this.dispose();
  };

  private handleError = async (error: Error) => {
    if (this.is_done) {
      return;
    }
    await this.channel.sendEvent({
      type: "ai_indicator.update",
      ai_state: "AI_STATE_ERROR",
      cid: this.message.cid,
      message_id: this.message.id,
    });
    await this.chatClient.partialUpdateMessage(this.message.id, {
      set: {
        text: error.message ?? "Error generating the message",
        message: error.toString(),
      },
    });
    await this.dispose();
  };

  // Method to set the prompt text
  setPrompt = (prompt: string) => {
    this.message_text = prompt;
  };
}
