// src/frontend/src/pages/private/DashboardPage.jsx
import React from "react";
import Sidebar from "../../components/private/Sidebar";
import AnalysisStatus from "../../components/private/AnalysisStatus";
import RecentResultsTable from "../../components/private/RecentResultsTable";
import StatsSummary from "../../components/private/StatsSummary";
import { useAuth } from "../../auth/AuthProvider";

const DashboardPage = () => {
  const { user } = useAuth();
  console.log("Usuario en contexto:", user);
  return (
    <div>
      <Sidebar />

      <main className="main-content p-4 p-md-5">
        <header className="mb-5 d-flex justify-content-between align-items-center">
          <div>
            <h1 className="h2">Panel de Control</h1>
            <p className="text-muted">
              Bienvenido {user ? `${user.nombres} ${user.apellidos}` : "Doctor"}
              .
            </p>
          </div>
        </header>

        <section className="mb-5">
          <AnalysisStatus />
        </section>

        <section className="mb-5">
          <RecentResultsTable />
        </section>

        <section>
          <StatsSummary />
        </section>
      </main>
    </div>
  );
};

export default DashboardPage;
