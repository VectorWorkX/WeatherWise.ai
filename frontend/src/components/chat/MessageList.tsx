import type { ChatMessage } from "../../types/chat";
import { useAutoScroll } from "../../hooks/useAutoScroll";
import MessageBubble from "./MessageBubble";

interface Props {
  messages: ChatMessage[];
  onApproval?: (approved: boolean) => void;
}

export default function MessageList({ messages, onApproval }: Props) {
  const scrollRef = useAutoScroll<HTMLDivElement>([messages]);

  return (
    <div
      ref={scrollRef}
      className="flex-1 overflow-y-auto px-4 py-6"
    >
      <div className="mx-auto flex max-w-3xl flex-col gap-6">
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} onApproval={onApproval} />
        ))}
      </div>
    </div>
  );
}
