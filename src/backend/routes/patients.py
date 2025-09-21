from flask import Blueprint, request, jsonify
from db import get_connection
from psycopg2.extras import RealDictCursor
from flask_jwt_extended import jwt_required

patients_bp = Blueprint("patients", __name__)

# =====================
# Crear paciente
# =====================
@patients_bp.route("/pacientes", methods=["POST"])
@jwt_required()
def create_patient():
    data = request.json
    nombres = data.get("nombres")
    apellidos = data.get("apellidos")
    genero = data.get("genero")
    fecha_nacimiento = data.get("fecha_nacimiento")
    edad = data.get("edad")
    dui = data.get("dui")
    id_municipio = data.get("id_municipio")

    if not all([nombres, apellidos, genero, fecha_nacimiento, edad, id_municipio]):
        return jsonify({"msg": "Faltan datos obligatorios"}), 400

    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute(
            """
            INSERT INTO pacientes (nombres, apellidos, genero, fecha_nacimiento, edad, dui, id_municipio)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id_paciente
            """,
            (nombres, apellidos, genero, fecha_nacimiento, edad, dui, id_municipio),
        )

        paciente_id = cur.fetchone()["id_paciente"]
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({"msg": "Paciente registrado con éxito", "id_paciente": paciente_id}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================
# Listar todos los pacientes
# =====================
@patients_bp.route("/pacientes", methods=["GET"])
@jwt_required()
def get_patients():
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            SELECT p.id_paciente, p.nombres, p.apellidos, p.genero, p.fecha_nacimiento,
                   p.edad, p.dui, m.nombre AS municipio
            FROM pacientes p
            JOIN municipios m ON p.id_municipio = m.id_municipio
            ORDER BY p.apellidos ASC
            """
        )
        pacientes = cur.fetchall()
        cur.close()
        conn.close()

        return jsonify({"pacientes": pacientes}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================
# Obtener paciente por ID
# =====================
@patients_bp.route("/pacientes/<uuid:id_paciente>", methods=["GET"])
@jwt_required()
def get_patient(id_paciente):
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            SELECT p.id_paciente, p.nombres, p.apellidos, p.genero, p.fecha_nacimiento,
                   p.edad, p.dui, m.nombre AS municipio
            FROM pacientes p
            JOIN municipios m ON p.id_municipio = m.id_municipio
            WHERE p.id_paciente = %s
            """,
            (str(id_paciente),),
        )
        paciente = cur.fetchone()
        cur.close()
        conn.close()

        if not paciente:
            return jsonify({"msg": "Paciente no encontrado"}), 404

        return jsonify({"paciente": paciente}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================
# Actualizar paciente
# =====================
@patients_bp.route("/pacientes/<uuid:id_paciente>", methods=["PUT"])
@jwt_required()
def update_patient(id_paciente):
    data = request.json
    campos = ["nombres", "apellidos", "genero", "fecha_nacimiento", "edad", "dui", "id_municipio"]

    # Construir query dinámicamente solo con los campos enviados
    updates = []
    valores = []
    for campo in campos:
        if campo in data:
            updates.append(f"{campo} = %s")
            valores.append(data[campo])

    if not updates:
        return jsonify({"msg": "No hay campos para actualizar"}), 400

    valores.append(str(id_paciente))

    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            f"""
            UPDATE pacientes
            SET {", ".join(updates)}, actualizado_en = NOW()
            WHERE id_paciente = %s
            RETURNING id_paciente
            """,
            valores,
        )

        if cur.rowcount == 0:
            conn.rollback()
            cur.close()
            conn.close()
            return jsonify({"msg": "Paciente no encontrado"}), 404

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({"msg": "Paciente actualizado con éxito"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================
# Eliminar paciente
# =====================
@patients_bp.route("/pacientes/<uuid:id_paciente>", methods=["DELETE"])
@jwt_required()
def delete_patient(id_paciente):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM pacientes WHERE id_paciente = %s", (str(id_paciente),))

        if cur.rowcount == 0:
            conn.rollback()
            cur.close()
            conn.close()
            return jsonify({"msg": "Paciente no encontrado"}), 404

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({"msg": "Paciente eliminado con éxito"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================
# Buscar paciente por DUI
# =====================
@patients_bp.route("/pacientes/buscar", methods=["GET"])
@jwt_required()
def buscar_paciente():
    dui = request.args.get("dui")
    if not dui:
        return jsonify({"msg": "El parámetro 'dui' es requerido"}), 400

    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            SELECT p.id_paciente, p.nombres, p.apellidos, p.genero, 
                   p.fecha_nacimiento, p.edad, p.dui, m.nombre AS municipio
            FROM pacientes p
            JOIN municipios m ON p.id_municipio = m.id_municipio
            WHERE p.dui = %s
            """,
            (dui,),
        )
        paciente = cur.fetchone()
        cur.close()
        conn.close()

        if not paciente:
            return jsonify({"msg": "Paciente no encontrado"}), 404

        return jsonify({"paciente": paciente}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
