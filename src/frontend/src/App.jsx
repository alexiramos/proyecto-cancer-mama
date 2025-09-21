// src/App.jsx
import React, { Children } from "react";
import { Routes, Route } from "react-router-dom";

import PublicLayout from "./assets/layouts/PublicLayout";
import PrivateLayout from "./assets/layouts/PrivateLayout";

// Páginas públicas
import HomePage from "./pages/public/HomePage";
import ContactPage from "./pages/public/ContactPage";
import ServicesPage from "./pages/public/ServicesPage";
import AboutPage from "./pages/public/AboutPage";
import LoginPage from "./pages/public/LoginPage";

// Página privada
import DashboardPage from "./pages/private/DashboardPage";
import ProtectedRoute from "./pages/private/ProtectedRoute";
import RegisterPage from "./pages/public/RegisterPage";
import PatientRegistration from "./pages/private/PatientRegistration";
import PatientList from "./pages/private/PatientList";
import AccountPage from "./pages/private/AccountPage";

function App() {
  return (
    <Routes>
      {/* Rutas públicas */}
      <Route element={<PublicLayout />}>
        <Route path="/" element={<HomePage />} />
        <Route path="/contact" element={<ContactPage />} />
        <Route path="/services" element={<ServicesPage />} />
        <Route path="/about" element={<AboutPage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
      </Route>

      {/* Rutas privadas */}
      {/*<Route element={<PrivateLayout />}>*/}
      <Route element={<ProtectedRoute allowedRoles={["medico"]} />}>
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/patientregistration" element={<PatientRegistration />} />
        <Route path="/patients" element={<PatientList />} />
        <Route path="/account" element={<AccountPage />} />
      </Route>

      {/* Ruta 404 */}
      <Route
        path="*"
        element={<div className="container py-5">Página no encontrada</div>}
      />
    </Routes>
  );
}

export default App;
