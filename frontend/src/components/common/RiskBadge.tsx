interface Props {
  risk: "Safe" | "Suspicious" | "Danger";
}

export default function RiskBadge({ risk }: Props) {
  const color =
    risk === "Safe"
      ? "bg-green-500"
      : risk === "Suspicious"
      ? "bg-yellow-500"
      : "bg-red-600";

  return (
    <span
      className={`text-white px-4 py-2 rounded-full ${color}`}
    >
      {risk}
    </span>
  );
}