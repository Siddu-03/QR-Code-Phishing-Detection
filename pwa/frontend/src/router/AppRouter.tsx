import { Routes, Route } from "react-router-dom";

import Navbar from "../components/layout/Navbar";
import Footer from "../components/layout/Footer";

import Home from "../pages/Home";
import Scanner from "../pages/Scanner";
import Upload from "../pages/Upload";
import Results from "../pages/Results";
import History from "../pages/History";
import Settings from "../pages/Settings";

export default function AppRouter() {
  return (
    <>
      <Navbar />

      <main className="min-h-screen">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/scanner" element={<Scanner />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/results" element={<Results />} />
          <Route path="/history" element={<History />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </main>

      <Footer />
    </>
  );
}