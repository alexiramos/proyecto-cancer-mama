// src/frontend/src/pages/public/RegisterPage.jsx
import React, { useState } from "react";
import { Link, Navigate } from "react-router-dom";
import { useAuth } from "../../auth/AuthProvider";

function RegisterPage() {
  const auth = useAuth();

  if (auth.isAuthenticated) {
    return <Navigate to="/dashboard" />;
  }

  const [nombres, setNombres] = useState("");
  const [apellidos, setApellidos] = useState("");
  const [email, setEmail] = useState("");
  const [pass, setPass] = useState("");
  const [pass2, setPass2] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);

  const validEmail = (v) => /\S+@\S+\.\S+/.test(v);
  const passwordsMatch = pass && pass2 && pass === pass2;

  const canSubmit =
    nombres.trim().length >= 2 &&
    apellidos.trim().length >= 2 &&
    validEmail(email) &&
    pass.length >= 6 &&
    passwordsMatch;

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!canSubmit) return;

    setSubmitting(true);
    setError(null);
    setSuccess(false);

    try {
      const response = await fetch("http://127.0.0.1:5000/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          nombres,
          apellidos,
          correo: email,
          password: pass,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        setSuccess(true);
        // ✅ Opcional: podrías loguear directamente después de registrarse
        // auth.login({ id_usuario: data.id_usuario, correo: email, rol: "medico" }, "");
        // Pero mejor dejamos que vaya al login.
      } else {
        setError(data.msg || "Error al registrar usuario");
      }
    } catch (err) {
      console.error(err);
      setError("Error de conexión con el servidor");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="container pb-5" style={{ paddingTop: "110px" }}>
      <h2 className="text-center fw-bold mb-4">Regístrate o Inicia Sesión</h2>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-4">
        <li className="nav-item">
          <span className="nav-link active">Registro</span>
        </li>
        <li className="nav-item">
          <Link className="nav-link" to="/login">
            Inicio de Sesión
          </Link>
        </li>
      </ul>

      <div className="row justify-content-center">
        <div className="col-12 col-md-8 col-lg-6">
          <div className="card border-0 shadow-sm">
            <div className="card-body p-4 p-lg-5">
              <form onSubmit={handleSubmit} noValidate>
                {/* Nombres */}
                <div className="form-floating mb-3">
                  <input
                    type="text"
                    className="form-control form-control-lg"
                    id="nombres"
                    placeholder="Nombres"
                    value={nombres}
                    onChange={(e) => setNombres(e.target.value)}
                    required
                  />
                  <label htmlFor="nombres" className="text-muted">
                    Nombres
                  </label>
                </div>

                {/* Apellidos */}
                <div className="form-floating mb-3">
                  <input
                    type="text"
                    className="form-control form-control-lg"
                    id="apellidos"
                    placeholder="Apellidos"
                    value={apellidos}
                    onChange={(e) => setApellidos(e.target.value)}
                    required
                  />
                  <label htmlFor="apellidos" className="text-muted">
                    Apellidos
                  </label>
                </div>

                {/* Correo electrónico */}
                <div className="form-floating mb-3">
                  <input
                    type="email"
                    className="form-control form-control-lg"
                    id="email"
                    placeholder="correo@ejemplo.com"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                  />
                  <label htmlFor="email" className="text-muted">
                    Correo electrónico
                  </label>
                </div>

                {/* Contraseña */}
                <div className="form-floating mb-3">
                  <input
                    type="password"
                    className="form-control form-control-lg"
                    id="password"
                    placeholder="Contraseña"
                    value={pass}
                    onChange={(e) => setPass(e.target.value)}
                    minLength={6}
                    required
                  />
                  <label htmlFor="password" className="text-muted">
                    Contraseña
                  </label>
                </div>

                {/* Confirmar contraseña */}
                <div className="form-floating mb-4">
                  <input
                    type="password"
                    className="form-control form-control-lg"
                    id="password2"
                    placeholder="Confirmar contraseña"
                    value={pass2}
                    onChange={(e) => setPass2(e.target.value)}
                    required
                  />
                  <label htmlFor="password2" className="text-muted">
                    Confirmar contraseña
                  </label>
                </div>

                {error && <div className="alert alert-danger">{error}</div>}
                {success && (
                  <div className="alert alert-success">
                    Registro exitoso 🎉 Ahora puedes{" "}
                    <Link to="/login">iniciar sesión</Link>.
                  </div>
                )}

                <button
                  type="submit"
                  className="btn btn-primary btn-lg w-100"
                  disabled={!canSubmit || submitting}>
                  {submitting ? "Registrando..." : "Registrarse"}
                </button>

                <p className="text-center mt-4 mb-0">
                  ¿Ya tienes una cuenta?{" "}
                  <Link to="/login" className="text-decoration-none">
                    Inicia sesión
                  </Link>
                </p>
              </form>
            </div>
          </div>
        </div>
      </div>

      <p className="text-center text-muted mt-3 small">&nbsp;</p>
    </div>
  );
}

export default RegisterPage;
