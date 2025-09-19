import React, { useState, useEffect } from "react";

const PatientRegistrationForm = () => {
  const [form, setForm] = useState({
    nombres: "",
    apellidos: "",
    genero: "",
    fecha_nacimiento: "",
    edad: "",
    dui: "",
    id_departamento: "",
    id_municipio: "",
  });

  const [departamentos, setDepartamentos] = useState([]);
  const [municipios, setMunicipios] = useState([]);

  // 🔹 Obtener departamentos desde el backend
  useEffect(() => {
    const token = localStorage.getItem("token");

    fetch("http://127.0.0.1:5000/api/departamentos", {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    })
      .then((res) => res.json())
      .then((data) => {
        console.log("DEBUG departamentos:", data);
        setDepartamentos(data.departamentos || []);
      })
      .catch((err) => console.error("Error cargando departamentos:", err));
  }, []);

  // 🔹 Cuando cambia el departamento, obtener municipios
  useEffect(() => {
    if (form.id_departamento) {
      const token = localStorage.getItem("token");

      fetch(`http://127.0.0.1:5000/api/municipios/${form.id_departamento}`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      })
        .then((res) => res.json())
        .then((data) => {
          console.log("DEBUG municipios:", data);
          setMunicipios(data.municipios || []);
        })
        .catch((err) => console.error("Error cargando municipios:", err));
    } else {
      setMunicipios([]);
    }
  }, [form.id_departamento]);

  // 🔹 Actualizar estado del form
  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  // 🔹 Enviar datos al backend
  const handleSubmit = async (e) => {
    e.preventDefault();

    const token = localStorage.getItem("token");
    const res = await fetch("http://127.0.0.1:5000/api/pacientes", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(form),
    });

    const data = await res.json();
    if (res.ok) {
      alert("Paciente registrado con éxito ✅");
      console.log(data);
    } else {
      alert("Error: " + (data.msg || data.error));
    }
  };

  return (
    <div className="container py-5">
      <h2 className="text-center fw-bold mb-4">Registro de Paciente</h2>

      <form onSubmit={handleSubmit} className="card p-4 shadow-sm">
        {/* Nombres */}
        <div className="form-floating mb-3">
          <input
            type="text"
            className="form-control"
            id="nombres"
            name="nombres"
            value={form.nombres}
            onChange={handleChange}
            placeholder="Nombre"
            required
          />
          <label htmlFor="nombres">Nombre</label>
        </div>

        {/* Apellidos */}
        <div className="form-floating mb-3">
          <input
            type="text"
            className="form-control"
            id="apellidos"
            name="apellidos"
            value={form.apellidos}
            onChange={handleChange}
            placeholder="Apellido"
            required
          />
          <label htmlFor="apellidos">Apellido</label>
        </div>

        {/* Género */}
        <div className="form-floating mb-3">
          <select
            className="form-select"
            id="genero"
            name="genero"
            value={form.genero}
            onChange={handleChange}
            required
          >
            <option value="">Seleccione...</option>
            <option value="F">Femenino</option>
            <option value="M">Masculino</option>
            <option value="Otro">Otro</option>
          </select>
          <label htmlFor="genero">Género</label>
        </div>

        {/* Fecha nacimiento */}
        <div className="form-floating mb-3">
          <input
            type="date"
            className="form-control"
            id="fecha_nacimiento"
            name="fecha_nacimiento"
            value={form.fecha_nacimiento}
            onChange={handleChange}
            required
          />
          <label htmlFor="fecha_nacimiento">Fecha de Nacimiento</label>
        </div>

        {/* Edad */}
        <div className="form-floating mb-3">
          <input
            type="number"
            className="form-control"
            id="edad"
            name="edad"
            value={form.edad}
            onChange={handleChange}
            required
          />
          <label htmlFor="edad">Edad</label>
        </div>

        {/* DUI */}
        <div className="form-floating mb-3">
          <input
            type="text"
            className="form-control"
            id="dui"
            name="dui"
            placeholder="12345678-9"
            value={form.dui}
            onChange={handleChange}
            pattern="^[0-9]{8}-[0-9]$"
          />
          <label htmlFor="dui">Número Único de Identidad (DUI)</label>
        </div>

        {/* Departamento */}
        <div className="form-floating mb-3">
          <select
            className="form-select"
            id="departamento"
            name="id_departamento"
            value={form.id_departamento}
            onChange={handleChange}
            required
          >
            <option value="">Seleccione...</option>
            {departamentos.map((d) => (
              <option key={d.id_departamento} value={d.id_departamento}>
                {d.nombre}
              </option>
            ))}
          </select>
          <label htmlFor="departamento">Departamento</label>
        </div>

        {/* Municipio */}
        <div className="form-floating mb-3">
          <select
            className="form-select"
            id="municipio"
            name="id_municipio"
            value={form.id_municipio}
            onChange={handleChange}
            required
          >
            <option value="">Seleccione...</option>
            {municipios.map((m) => (
              <option key={m.id_municipio} value={m.id_municipio}>
                {m.nombre}
              </option>
            ))}
          </select>
          <label htmlFor="municipio">Municipio</label>
        </div>

        {/* Botón */}
        <button type="submit" className="btn btn-primary btn-lg w-100">
          Guardar Paciente
        </button>
      </form>
    </div>
  );
};

export default PatientRegistrationForm;
