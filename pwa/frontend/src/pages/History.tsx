import RiskBadge from "../components/common/RiskBadge";

const history = [
  {
    id: 1,
    url: "https://google.com",
    risk: "Safe",
    date: "18 Jul 2026",
  },
  {
    id: 2,
    url: "http://fake-bank-login.xyz",
    risk: "Danger",
    date: "17 Jul 2026",
  },
  {
    id: 3,
    url: "https://university.edu",
    risk: "Safe",
    date: "16 Jul 2026",
  },
] as const;

export default function History() {
  return (
    <div className="max-w-5xl mx-auto py-10 px-6">
      <h1 className="text-4xl font-bold mb-8">Scan History</h1>

      <div className="space-y-4">
        {history.map((item) => (
          <div
            key={item.id}
            className="bg-white shadow rounded-xl p-5 flex justify-between items-center"
          >
            <div>
              <p className="font-semibold break-all">{item.url}</p>
              <p className="text-gray-500 text-sm">{item.date}</p>
            </div>

            <RiskBadge risk={item.risk} />
          </div>
        ))}
      </div>
    </div>
  );
}