import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, RouterProvider } from 'react-router-dom'
// 5. COMPONENTE PRINCIPAL
import App from './App.jsx'
// 4. ESTILOS PROPIOS (si existen)
import './index.css'

import 'bootstrap/dist/css/bootstrap.min.css';
// 1. BOOTSTRAP PERSONALIZADO (SCSS sobrescribe variables de Bootstrap)
import './styles/bootstrap-custom.scss'

// 2. BOOTSTRAP JS (para que funcionen dropdowns, modals, etc.)
import 'bootstrap/dist/js/bootstrap.bundle.min.js'

// 3. ICONOS DE BOOTSTRAP
import 'bootstrap-icons/font/bootstrap-icons.css'
import { AuthProvider } from './auth/AuthProvider.jsx'


// 6. RENDER DE LA APLICACIÓN
const root = createRoot(document.getElementById('root'));
root.render(
  <StrictMode>
    <AuthProvider>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </AuthProvider>
  </StrictMode>
);

