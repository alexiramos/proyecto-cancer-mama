from flask import Blueprint, jsonify
from db import get_connection
from psycopg2.extras import RealDictCursor
from flask_jwt_extended import jwt_required

geo_bp = Blueprint("geo", __name__)

# =====================
# Listar departamentos
# =====================
@geo_bp.route("/departamentos", methods=["GET"])
@jwt_required()   # proteger si quieres que solo usuarios autenticados accedan
def get_departamentos():
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT id_departamento, nombre FROM departamentos ORDER BY nombre ASC")
        departamentos = cur.fetchall()
        cur.close()
        conn.close()

        return jsonify({"departamentos": departamentos}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================
# Listar municipios por departamento
# =====================
@geo_bp.route("/municipios/<uuid:id_departamento>", methods=["GET"])
@jwt_required()
def get_municipios(id_departamento):
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT id_municipio, nombre FROM municipios WHERE id_departamento = %s ORDER BY nombre ASC",
            (str(id_departamento),),
        )
        municipios = cur.fetchall()
        cur.close()
        conn.close()

        return jsonify({"municipios": municipios}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
