// src/frontend/src/pages/private/AccountPage.jsx
import React from "react";
import { Container, Row, Col, Image, Form, Button } from "react-bootstrap";
import Sidebar from "../../components/private/Sidebar";

export default function AccountPage() {
  return (
    <Container
      fluid
      style={{
        fontFamily: '"Public Sans", sans-serif',
        backgroundColor: "#f8f9fa",
      }}>
      <Row className="min-vh-100 bg-white">
        <Col md={4} lg={3} className="p-0">
          <Sidebar />
        </Col>
        <Col md={8} lg={9} className="py-4 px-md-5">
          <h1
            className="fw-bold mb-4"
            style={{ fontSize: "32px", color: "#111218" }}>
            Información de la Cuenta
          </h1>

          <div className="text-center mb-5">
            <Image
              src="https://placehold.co/200x200/607AFB/FFFFFF?text=CR"
              roundedCircle
              style={{
                width: "128px",
                height: "128px",
                objectFit: "cover",
                border: "2px solid #f0f1f5",
              }}
              className="mb-3 shadow-sm"
            />
            <h2 className="fw-bold fs-4" style={{ color: "#111218" }}>
              Carlos Ramirez
            </h2>
            <p className="mb-1" style={{ color: "#5f668c" }}>
              Radiología
            </p>
            <p className="text-muted small">
              Haga clic para subir una nueva imagen
            </p>
          </div>

          <div style={{ maxWidth: "480px" }} className="mx-auto mx-md-0">
            <Form>
              {[
                "Nombre",
                "Apellido",
                "Correo Electrónico",
                "Número de Teléfono",
                "Especialidad",
              ].map((label, idx) => (
                <Form.Group
                  className="mb-3"
                  controlId={`form${label.replace(/\s/g, "")}`}
                  key={idx}>
                  <Form.Label>{label}</Form.Label>
                  <Form.Control
                    type={idx === 2 ? "email" : "text"}
                    size="lg"
                    placeholder={label}
                  />
                </Form.Group>
              ))}

              <h3 className="fw-bold fs-5 mt-5 mb-3">Cambiar Contraseña</h3>

              {[
                "Contraseña Actual",
                "Nueva Contraseña",
                "Confirmar Nueva Contraseña",
              ].map((label, idx) => (
                <Form.Group
                  className="mb-3"
                  controlId={`formPassword${idx}`}
                  key={idx}>
                  <Form.Label>{label}</Form.Label>
                  <Form.Control
                    type="password"
                    placeholder={`Ingresa ${label.toLowerCase()}`}
                    size="lg"
                  />
                </Form.Group>
              ))}

              <Button
                variant="primary"
                type="submit"
                className="mt-3 fw-bold"
                size="lg"
                style={{ backgroundColor: "#607afb", borderColor: "#607afb" }}>
                Guardar Cambios
              </Button>
            </Form>
          </div>
        </Col>
      </Row>
    </Container>
  );
}
