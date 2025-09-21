// src/frontend/src/pages/public/ContactPage.jsx
import React from "react";

const ContactPage = () => {
  return (
    <div className="container" style={{ maxWidth: "700px" }}>
      <h1 className="page-title">Contáctanos</h1>
      <p className="page-subtitle">
        Si tienes preguntas sobre nuestros servicios de diagnóstico automatizado
        de cáncer o necesitas soporte técnico, no dudes en contactarnos. Estamos
        aquí para ayudarte.
      </p>

      <div className="contact-info-section">
        <h2 className="section-heading">Información de Contacto</h2>
        <div className="mb-3">
          <p className="info-label mb-1">Correo Electrónico</p>
          <p className="info-value mb-0">contacto@l.com</p>
        </div>
        <div>
          <p className="info-label mb-1">Dirección</p>
          <p className="info-value mb-0">
            Colonia Médica, San Salvador, El Salvador
          </p>
        </div>
      </div>

      <div className="contact-form-section">
        <h2 className="section-heading">Formulario de Contacto</h2>
        <form>
          <div className="mb-3">
            <label htmlFor="name" className="form-label">
              Nombre
            </label>
            <input
              type="text"
              className="form-control"
              id="name"
              placeholder="Tu nombre"
            />
          </div>
          <div className="mb-3">
            <label htmlFor="email" className="form-label">
              Correo Electrónico
            </label>
            <input
              type="email"
              className="form-control"
              id="email"
              placeholder="Tu correo electrónico"
            />
          </div>
          <div className="mb-3">
            <label htmlFor="subject" className="form-label">
              Asunto
            </label>
            <input
              type="text"
              className="form-control"
              id="subject"
              placeholder="Asunto del mensaje"
            />
          </div>
          <div className="mb-4">
            <label htmlFor="message" className="form-label">
              Mensaje
            </label>
            <textarea className="form-control" id="message" rows="5"></textarea>
          </div>
          <button type="submit" className="btn btn-primary">
            Enviar Mensaje
          </button>
        </form>
      </div>
    </div>
  );
};

export default ContactPage;
