// src/frontend/src/pages/public/AboutPage.jsx
import React from "react";

const AboutPage = () => {
  // Reemplazando las importaciones locales con URLs de marcador de posición para solucionar el error de compilación.
  const radiologaImg = "/src/assets/images/radiologa.png";
  const oncologoImg = "/src/assets/images/Oncólogo .png";
  const ingenieraImg = "/src/assets/images/ingeniera.png";

  return (
    <div className="container">
      <section className="page-section">
        <h1 className="section-title">Acerca de HealthVisionSV</h1>
        <p className="section-text">
          HealthVisionSV es una iniciativa innovadora en El Salvador, dedicada
          al desarrollo de un sistema web para el diagnóstico automatizado de
          cáncer a partir de imágenes radiológicas. Nuestra misión es mejorar la
          precisión y eficiencia del diagnóstico médico, facilitando el acceso a
          tecnología de punta para profesionales de la salud en todo el país.
        </p>
      </section>

      <section className="page-section">
        <h2 className="section-subtitle">Nuestros Objetivos</h2>
        <p className="section-text">
          Buscamos transformar el panorama del diagnóstico de cáncer en El
          Salvador, enfocándonos en:
        </p>
        <ul className="objective-list mt-4">
          <li className="objective-item">
            Aumentar la precisión diagnóstica mediante el uso de algoritmos
            avanzados de inteligencia artificial.
          </li>
          <li className="objective-item">
            Reducir los tiempos de espera para obtener resultados, permitiendo
            intervenciones médicas más tempranas.
          </li>
          <li className="objective-item">
            Democratizar el acceso a tecnología de diagnóstico de alta calidad,
            independientemente de la ubicación geográfica del paciente o del
            centro médico.
          </li>
        </ul>
      </section>

      <section className="page-section">
        <h2 className="section-subtitle">El Equipo</h2>
        <p className="section-text">
          Nuestro equipo está compuesto por expertos en radiología, oncología,
          ingeniería de software e inteligencia artificial, unidos por la pasión
          de innovar en el cuidado de la salud. Colaboramos estrechamente con
          instituciones médicas y académicas líderes para asegurar la validez y
          aplicabilidad de nuestro sistema.
        </p>
        <div className="row mt-5">
          <div className="col-md-4 team-member">
            <img
              src={radiologaImg}
              className="team-member-img"
              alt="Radióloga Oncologa"
            />
            <h3 className="team-member-name">Dra. Sofía Mendoza</h3>
            <p className="team-member-title">Radióloga Oncóloga</p>
          </div>
          <div className="col-md-4 team-member">
            <img src={oncologoImg} className="team-member-img" alt="Clínico" />
            <h3 className="team-member-name">Dr. Ricardo Silva</h3>
            <p className="team-member-title">Clínico</p>
          </div>
          <div className="col-md-4 team-member">
            <img
              src={ingenieraImg}
              className="team-member-img"
              alt="Ingeniera de Software"
            />
            <h3 className="team-member-name">Ing. Elena Vargas</h3>
            <p className="team-member-title">Ingeniera de Software</p>
          </div>
        </div>
      </section>

      <section className="page-section">
        <h2 className="section-subtitle">Investigación y Desarrollo</h2>
        <p className="section-text">
          El sistema HealthVision es el resultado de años de investigación y
          desarrollo, utilizando técnicas avanzadas de aprendizaje profundo para
          analizar imágenes radiológicas. Nos enfocamos en la detección temprana
          de diferentes tipos de cáncer, mejorando continuamente nuestros
          algoritmos con datos clínicos reales.
        </p>
      </section>

      <section className="page-section">
        <h2 className="section-subtitle">Valores</h2>
        <p className="section-text">
          Nuestros valores fundamentales son la precisión, la integridad, la
          innovación y el compromiso con la salud de la comunidad salvadoreña.
          Nos esforzamos por mantener los más altos estándares éticos y
          profesionales en todas nuestras actividades.
        </p>
      </section>
    </div>
  );
};

export default AboutPage;
