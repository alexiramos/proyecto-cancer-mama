from flask import Blueprint, request, jsonify
from werkzeug.security import check_password_hash, generate_password_hash
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from db import get_connection
from psycopg2.extras import RealDictCursor

auth_bp = Blueprint("auth", __name__)

# =====================
# Registro (con rol médico por defecto)
# =====================
@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.json
    nombres = data.get("nombres")
    apellidos = data.get("apellidos")
    correo = data.get("correo")
    password = data.get("password")

    if not all([nombres, apellidos, correo, password]):
        return jsonify({"msg": "Faltan datos"}), 400

    hashed_pw = generate_password_hash(password)

    try:
        conn = get_connection()
        cur = conn.cursor()

        # Rol médico por defecto
        cur.execute("SELECT id_rol FROM roles WHERE nombre = %s", ("medico",))
        rol = cur.fetchone()
        if not rol:
            return jsonify({"msg": "No existe el rol 'medico' en la BD"}), 500

        id_rol = rol[0]

        # Insertar usuario
        cur.execute(
            """
            INSERT INTO usuarios (id_rol, nombres, apellidos, correo, contrasena_hash)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id_usuario
            """,
            (id_rol, nombres, apellidos, correo, hashed_pw),
        )

        user_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({"msg": "Usuario registrado con éxito", "id_usuario": user_id}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================
# Login
# =====================
@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.json
    correo = data.get("correo")
    password = data.get("password")

    if not correo or not password:
        return jsonify({"msg": "Correo y contraseña son obligatorios"}), 400

    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Buscar usuario y rol
        cur.execute(
            """
            SELECT u.id_usuario, u.nombres, u.apellidos, u.contrasena_hash, r.nombre AS rol
            FROM usuarios u
            JOIN roles r ON r.id_rol = u.id_rol
            WHERE u.correo = %s
            """,
            (correo,),
        )
        user = cur.fetchone()
        cur.close()
        conn.close()

        if not user:
            return jsonify({"msg": "Usuario no encontrado"}), 404

        # Verificar contraseña
        if not check_password_hash(user["contrasena_hash"], password):
            return jsonify({"msg": "Credenciales inválidas"}), 401

        # Crear token usando solo id_usuario como identity
        access_token = create_access_token(identity=str(user["id_usuario"]))

        return jsonify({
            "msg": "Login exitoso",
            "access_token": access_token,
            "usuario": {
                "id_usuario": str(user["id_usuario"]),
                "nombres": user["nombres"],
                "apellidos": user["apellidos"],
                "correo": correo,
                "rol": user["rol"]
            }
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================
# Obtener usuario autenticado (/me)
# =====================
@auth_bp.route("/me", methods=["GET"])
@jwt_required()
def me():
    try:
        # Recuperar el id_usuario del token
        user_id = get_jwt_identity()

        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute(
            """
            SELECT u.id_usuario, u.nombres, u.apellidos, u.correo, r.nombre AS rol
            FROM usuarios u
            JOIN roles r ON r.id_rol = u.id_rol
            WHERE u.id_usuario = %s
            """,
            (user_id,),
        )
        user = cur.fetchone()
        cur.close()
        conn.close()

        if not user:
            return jsonify({"msg": "Usuario no encontrado"}), 404

        return jsonify({"usuario": user}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
