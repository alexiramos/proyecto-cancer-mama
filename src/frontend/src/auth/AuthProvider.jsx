import { createContext, useContext, useState, useEffect } from "react";

const AuthContext = createContext({
  isAuthenticated: false,
  user: null,
  token: null,
  login: () => {},
  logout: () => {},
  loading: true, // 👈 nuevo estado
});

export function AuthProvider({ children }) {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [loading, setLoading] = useState(true); // 👈 inicia en "cargando"

  // Recuperar sesión al iniciar
  useEffect(() => {
    const savedToken = localStorage.getItem("token");

    if (savedToken) {
      fetch("http://127.0.0.1:5000/auth/me", {
        headers: {
          Authorization: `Bearer ${savedToken}`,
        },
      })
        .then((res) => res.json())
        .then((data) => {
          if (data.usuario) {
            setUser(data.usuario);
            setToken(savedToken);
            setIsAuthenticated(true);
            localStorage.setItem("user", JSON.stringify(data.usuario));
          } else {
            logout();
          }
        })
        .catch(() => logout())
        .finally(() => setLoading(false)); // 👈 terminamos de verificar
    } else {
      setLoading(false);
    }
  }, []);

  const login = (userData, token) => {
    setUser(userData);
    setToken(token);
    setIsAuthenticated(true);

    localStorage.setItem("token", token);
    localStorage.setItem("user", JSON.stringify(userData));
  };

  const logout = () => {
    setUser(null);
    setToken(null);
    setIsAuthenticated(false);

    localStorage.removeItem("token");
    localStorage.removeItem("user");
  };

  return (
    <AuthContext.Provider
      value={{ isAuthenticated, user, token, login, logout, loading }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
