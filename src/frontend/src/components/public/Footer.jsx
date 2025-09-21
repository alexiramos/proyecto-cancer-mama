// src/frontend/src/components/Footer.jsx
import React from "react";

const Footer = () => {
  return (
    <footer className="footer-custom bg-white mt-auto">
      <div className="container py-5">
        <div className="d-flex flex-column align-items-center gap-4">
          {/* Enlaces de navegación */}
          <nav
            className="nav flex-wrap justify-content-center justify-content-md-around w-100"
            style={{ gap: "1rem" }}>
            <a className="nav-link" href="#">
              Política de Privacidad
            </a>
            <a className="nav-link" href="#">
              Términos de Servicio
            </a>
            <a className="nav-link" href="/contact">
              Contacto
            </a>
          </nav>

          {/* Iconos de redes sociales */}
          <div className="d-flex justify-content-center gap-4">
            <a href="#" className="social-icon">
              <i className="bi bi-twitter-x"></i>
            </a>
            <a href="#" className="social-icon">
              <i className="bi bi-facebook"></i>
            </a>
            <a href="#" className="social-icon">
              <i className="bi bi-instagram"></i>
            </a>
          </div>

          {/* Texto de copyright */}
          <p className="mb-0">
            © 2025 <span className="fw-bold">HealthVisionSV.</span> Todos los
            derechos reservados.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
