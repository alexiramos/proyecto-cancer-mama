// src/components/private/Sidebar.jsx
import React from "react";
import { NavLink } from "react-router-dom";
import { useAuth } from "../../auth/AuthProvider"; // 👈 importa el hook
import { useNavigate } from "react-router-dom";

const Sidebar = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate("/login"); // 👈 redirigir al login después de cerrar sesión
  };
  return (
    <nav className="sidebar d-flex flex-column">
      <div className="sidebar-header p-4">
        <h5 className="text-white">Diagnóstico SV</h5>
        <small className="text-white-50">v1.0</small>
      </div>

      <ul className="nav flex-column p-3">
        <li className="nav-item">
          <NavLink
            to="/dashboard"
            end
            className={({ isActive }) =>
              `nav-link ${isActive ? "active" : ""}`
            }>
            <i className="bi bi-house-door-fill me-2"></i>
            Inicio
          </NavLink>
        </li>

        <li className="nav-item">
          <NavLink
            to="/patientregistration"
            className={({ isActive }) =>
              `nav-link ${isActive ? "active" : ""}`
            }>
            <i className="bi bi-person-plus-fill me-2"></i>
            Registro de paciente
          </NavLink>
        </li>

        <li className="nav-item">
          <NavLink
            to="/patients"
            className={({ isActive }) =>
              `nav-link ${isActive ? "active" : ""}`
            }>
            <i className="bi bi-people-fill me-2"></i>
            Lista de Pacientes
          </NavLink>
        </li>

        <li className="nav-item">
          <NavLink
            to="/analisis"
            className={({ isActive }) =>
              `nav-link ${isActive ? "active" : ""}`
            }>
            <i className="bi bi-plus-circle-fill me-2"></i>
            Nuevo Análisis
          </NavLink>
        </li>

        <li className="nav-item">
          <NavLink
            to="/historial"
            className={({ isActive }) =>
              `nav-link ${isActive ? "active" : ""}`
            }>
            <i className="bi bi-clock-history me-2"></i>
            Historial
          </NavLink>
        </li>

        <li className="nav-item">
          <NavLink
            to="/account"
            className={({ isActive }) =>
              `nav-link ${isActive ? "active" : ""}`
            }>
            <i className="bi bi-person-circle me-2"></i>
            Cuenta
          </NavLink>
        </li>
      </ul>

      <div className="mt-auto p-3">
        {/* Botón logout con estilo de nav-link */}
        <button
          className="nav-link text-danger border-0 bg-transparent"
          onClick={handleLogout}>
          <i className="bi bi-box-arrow-right me-2"></i>
          Cerrar Sesión
        </button>
      </div>
    </nav>
  );
};

export default Sidebar;
