// src/frontend/src/components/BenefitsSection.jsx

import React from "react";
import { Card } from "react-bootstrap";

const BenefitsSection = () => {
  return (
    <div className="benefits-section py-3">
      <div className="container-xxl">
        <h2 className="text-start fw-bold mb-3 section-title">
          Beneficios para el Sistema de Salud
        </h2>
        <p className="text-start mb-4">
          HealthVisionSV ofrece múltiples ventajas para el sistema de salud
          salvadoreño:
        </p>
        <div className="row justify-content-center g-4">
          <div className="col-md-4">
            <Card className="benefit-card h-100 text-center p-4">
              <div className="mb-3">
                <i
                  className="bi bi-search-heart"
                  style={{ fontSize: "2rem" }}></i>
              </div>
              <Card.Body>
                <Card.Title className="fw-bold">Precisión Mejorada</Card.Title>
                <Card.Text>
                  Utilizamos algoritmos de inteligencia artificial de última
                  generación para analizar imágenes radiológicas con alta
                  precisión, reduciendo errores y mejorando la confianza en los
                  resultados.
                </Card.Text>
              </Card.Body>
            </Card>
          </div>
          <div className="col-md-4">
            <Card className="benefit-card h-100 text-center p-4">
              <div className="mb-3">
                <i
                  className="bi bi-clock-history"
                  style={{ fontSize: "2rem" }}></i>
              </div>
              <Card.Body>
                <Card.Title className="fw-bold">
                  Eficiencia en el Diagnóstico
                </Card.Title>
                <Card.Text>
                  Nuestro sistema automatiza tareas repetitivas y acelera el
                  proceso de diagnóstico, permitiendo a los médicos enfocarse en
                  la atención al paciente y la toma de decisiones clínicas.
                </Card.Text>
              </Card.Body>
            </Card>
          </div>
          <div className="col-md-4">
            <Card className="benefit-card h-100 text-center p-4">
              <div className="mb-3">
                <i className="bi bi-heart" style={{ fontSize: "2rem" }}></i>
              </div>
              <Card.Body>
                <Card.Title className="fw-bold">Acceso Equitativo</Card.Title>
                <Card.Text>
                  HealthVision facilita el acceso a diagnósticos de calidad en
                  todo el país, incluyendo zonas rurales y remotas,
                  contribuyendo a la equidad en la atención médica.
                </Card.Text>
              </Card.Body>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BenefitsSection;
