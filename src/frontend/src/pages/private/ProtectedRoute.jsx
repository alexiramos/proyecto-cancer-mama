import { Outlet, Navigate } from "react-router-dom";
import { useAuth } from "../../auth/AuthProvider";

function ProtectedRoute() {
    const auth = useAuth();
    //const [isAuth, setIsAuth] = useState(false)

    return auth.isAuthenticated ? <Outlet /> : <Navigate to="/" />
}

export default ProtectedRoute;