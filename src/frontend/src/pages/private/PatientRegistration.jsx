import React from "react";
import Sidebar from "../../components/private/Sidebar";
import PatientRegistrationForm from "../../components/private/PatientRegistrationForm";

const PatientRegistration = () => {
  return (
    <div className="d-flex">
      {/* Sidebar fijo */}
      <Sidebar />

      {/* Contenido principal */}
      <main className="main-content p-4 p-md-5 flex-grow-1">
        <h1 className="h2 mb-4">Registro de Paciente</h1>
        <PatientRegistrationForm />
      </main>
    </div>
  );
};

export default PatientRegistration;
