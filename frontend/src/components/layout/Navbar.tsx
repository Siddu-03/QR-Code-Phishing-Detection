import { Link, NavLink } from "react-router-dom";
import { FaShieldAlt } from "react-icons/fa";

export default function Navbar() {
  const navLinkClass = ({ isActive }: { isActive: boolean }) =>
    isActive
      ? "text-blue-600 font-semibold"
      : "text-gray-600 hover:text-blue-600";

  return (
    <nav className="bg-white shadow-md">
      <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">

        <Link to="/" className="flex items-center gap-2">
          <FaShieldAlt className="text-blue-600 text-2xl" />
          <span className="text-xl font-bold">QR Shield</span>
        </Link>

        <div className="flex gap-6">
          <NavLink to="/" className={navLinkClass}>Home</NavLink>
          <NavLink to="/scanner" className={navLinkClass}>Scanner</NavLink>
          <NavLink to="/upload" className={navLinkClass}>Upload</NavLink>
          <NavLink to="/history" className={navLinkClass}>History</NavLink>
          <NavLink to="/settings" className={navLinkClass}>Settings</NavLink>
        </div>

      </div>
    </nav>
  );
}