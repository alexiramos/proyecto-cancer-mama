// src/frontend/src/pages/public/LoginPage.jsx
import React, { useState } from "react";
import { Link, Navigate } from "react-router-dom";
import { useAuth } from "../../auth/AuthProvider";

function LoginPage() {
  const auth = useAuth();

  if (auth.isAuthenticated) {
    return <Navigate to="/dashboard" />;
  }

  const [email, setEmail] = useState("");
  const [pass, setPass] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const response = await fetch("http://localhost:5000/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ correo: email, password: pass }),
      });

      const data = await response.json();

      if (response.ok) {
        const usuario = data.usuario;
        // Aseguramos que incluya nombres y apellidos
        auth.login(
          {
            id_usuario: usuario.id_usuario,
            nombres: usuario.nombres,
            apellidos: usuario.apellidos,
            correo: usuario.correo,
            rol: usuario.rol,
          },
          data.access_token
        );
      } else {
        setError(data.msg || "Error en el login");
      }
    } catch (err) {
      console.error(err);
      setError("Error de conexión con el servidor");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container pb-5" style={{ paddingTop: "110px" }}>
      <h2 className="text-center fw-bold mb-4">Regístrate o Inicia Sesión</h2>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-4">
        <li className="nav-item">
          <Link className="nav-link" to="/register">
            Registro
          </Link>
        </li>
        <li className="nav-item">
          <span className="nav-link active">Inicio de Sesión</span>
        </li>
      </ul>

      <div className="row justify-content-center">
        <div className="col-12 col-md-8 col-lg-6">
          <div className="card border-0 shadow-sm">
            <div className="card-body p-4 p-lg-5">
              <form onSubmit={handleSubmit} noValidate>
                {/* Usuario / correo */}
                <div className="form-floating mb-3">
                  <input
                    type="email"
                    className="form-control form-control-lg"
                    id="email"
                    placeholder="nombre@correo.com"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                  />
                  <label htmlFor="email" className="text-muted">
                    Nombre de usuario o correo electrónico
                  </label>
                </div>

                {/* Contraseña */}
                <div className="form-floating mb-2">
                  <input
                    type="password"
                    className="form-control form-control-lg"
                    id="password"
                    placeholder="Contraseña"
                    value={pass}
                    onChange={(e) => setPass(e.target.value)}
                    required
                  />
                  <label htmlFor="password" className="text-muted">
                    Contraseña
                  </label>
                </div>

                {error && <div className="alert alert-danger">{error}</div>}

                {/* Botón */}
                <button
                  type="submit"
                  className="btn btn-primary btn-lg w-100"
                  disabled={loading}>
                  {loading ? "Ingresando..." : "Iniciar Sesión"}
                </button>

                {/* Registro link */}
                <p className="text-center mt-4 mb-0">
                  ¿No tienes una cuenta?{" "}
                  <Link to="/register" className="text-decoration-none">
                    Regístrate
                  </Link>
                </p>
              </form>
            </div>
          </div>
        </div>
      </div>

      <p className="text-center text-muted mt-3 small">
        Activa notificaciones en tu perfil para recibir alertas de resultados.
      </p>
    </div>
  );
}

export default LoginPage;
