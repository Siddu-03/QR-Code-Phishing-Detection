interface ErrorCardProps {
  message: string;
}

export default function ErrorCard({ message }: ErrorCardProps) {
  return (
    <div className="bg-red-100 border border-red-400 text-red-700 p-4 rounded-xl my-4">
      <h2 className="font-bold mb-2">Error</h2>
      <p>{message}</p>
    </div>
  );
}