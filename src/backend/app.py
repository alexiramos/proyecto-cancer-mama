from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from dotenv import load_dotenv
import os

def create_app():
    load_dotenv()
    app = Flask(__name__)
    CORS(app)

    app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "supersecreto")
    JWTManager(app)

    # 👇 IMPORTANTE: importar y registrar el blueprint
    from routes.auth import auth_bp
    app.register_blueprint(auth_bp, url_prefix="/auth")

    from routes.patients import patients_bp
    app.register_blueprint(patients_bp, url_prefix="/api")

    from routes.geo import geo_bp
    app.register_blueprint(geo_bp, url_prefix="/api")


    @app.route("/")
    def home():
        return {"msg": "API Flask funcionando 🚀"}

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5000)
