// src/frontend/src/components/Navigation.jsx
import React from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import { Link } from "react-router-dom"; // Importa Link

const Navigation = () => {
  return (
    <nav className="navbar navbar-expand-lg navbar-light bg-custom fixed-top">
      <div className="container-fluid">
        <Link className="navbar-brand" to="/">
          HealthVisionSV
        </Link>
        <div className="collapse navbar-collapse">
          <ul className="navbar-nav ms-auto mb-2 mb-lg-0">
            <li className="nav-item">
              <Link className="nav-link" to="/">
                Inicio
              </Link>
            </li>
            <li className="nav-item">
              <Link className="nav-link" to="/about">
                Acerca de
              </Link>
            </li>
            <li className="nav-item">
              <Link className="nav-link" to="/services">
                Servicios
              </Link>
            </li>
            <li className="nav-item">
              <Link className="nav-link" to="/contact">
                Contacto
              </Link>
            </li>
            <li className="nav-item">
              <Link className="btn btn-primary ms-3" to="/login">
                Iniciar Sesión
              </Link>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
