// src/frontend/src/pages/private/PatientList.jsx
import React, { useState, useEffect } from "react";
import Sidebar from "../../components/private/Sidebar";

const PatientList = () => {
  const [pacientes, setPacientes] = useState([]);
  const [dui, setDui] = useState("");
  const [loading, setLoading] = useState(false);

  const token = localStorage.getItem("token");

  // 🔹 Cargar todos los pacientes al inicio
  useEffect(() => {
    fetch("http://127.0.0.1:5000/api/pacientes", {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => res.json())
      .then((data) => setPacientes(data.pacientes || []))
      .catch((err) => console.error("Error cargando pacientes:", err));
  }, [token]);

  // 🔹 Buscar paciente por DUI
  const handleSearch = () => {
    if (!dui.trim()) return;

    setLoading(true);
    fetch(`http://127.0.0.1:5000/api/pacientes/buscar?dui=${dui}`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.paciente) {
          setPacientes([data.paciente]); // mostrar solo ese paciente
        } else {
          setPacientes([]);
          alert(data.msg || "Paciente no encontrado");
        }
      })
      .catch((err) => console.error("Error en búsqueda:", err))
      .finally(() => setLoading(false));
  };

  return (
    <div className="d-flex">
      {/* Sidebar */}
      <Sidebar />

      {/* Contenido principal */}
      <main className="main-content p-4 p-md-5 flex-grow-1">
        <h2 className="fw-bold mb-4">Listado de Pacientes</h2>

        {/* Buscador por DUI */}
        <div className="input-group mb-4">
          <input
            type="text"
            className="form-control"
            placeholder="Buscar por DUI (Ej: 12345678-9)"
            value={dui}
            onChange={(e) => setDui(e.target.value)}
          />
          <button className="btn btn-primary" onClick={handleSearch}>
            Buscar
          </button>
        </div>

        {/* Tabla */}
        <div className="table-responsive">
          <table className="table table-striped table-hover">
            <thead className="table-dark">
              <tr>
                <th>Nombre</th>
                <th>Apellido</th>
                <th>Género</th>
                <th>Edad</th>
                <th>DUI</th>
                <th>Municipio</th>
              </tr>
            </thead>
            <tbody>
              {pacientes.length > 0 ? (
                pacientes.map((p) => (
                  <tr key={p.id_paciente}>
                    <td>{p.nombres}</td>
                    <td>{p.apellidos}</td>
                    <td>{p.genero}</td>
                    <td>{p.edad}</td>
                    <td>{p.dui}</td>
                    <td>{p.municipio}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="6" className="text-center text-muted">
                    {loading ? "Buscando..." : "No hay pacientes"}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </main>
    </div>
  );
};

export default PatientList;
