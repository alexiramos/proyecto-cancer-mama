// src/pages/public/ServicesPage.jsx
import React from "react";

const ServicesPage = () => {
  // Reemplazar con la imagen de los assets
  const mammographyImage = "/src/assets/images/nuestroservicios.png";

  return (
    <div className="container">
      <h1 className="page-title">Nuestros Servicios</h1>

      <section className="service-section">
        <h2 className="service-title">Mamografías</h2>
        <p className="service-description mt-3">
          Detección precisa de anomalías en mamografías con IA. Nuestro sistema
          analiza imágenes de alta resolución para identificar
          microcalcificaciones y otras señales tempranas de cáncer de mama.
        </p>
        <img
          src={mammographyImage}
          className="img-fluid shadow-sm service-image mx-auto d-block"
          alt="Análisis de mamografía"
        />
      </section>

      <section className="service-section">
        <h2 className="service-title">Tomografías Computarizadas</h2>
        <p className="service-description mt-3">
          Análisis detallado de tomografías para identificar posibles tumores.
          Nuestro sistema examina cortes transversales del cuerpo para detectar
          anomalías en órganos y tejidos, ofreciendo una visión completa para un
          diagnóstico preciso.
        </p>
      </section>
    </div>
  );
};

export default ServicesPage;
