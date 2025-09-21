// src/frontend/src/components/HeroBanner.jsx
import React from "react";

import heroBannerImage1 from "../../assets/images/hero_banner01.png";
import heroBannerImage2 from "../../assets/images/hero_banner02.png";
import heroBannerImage3 from "../../assets/images/hero_banner03.png";

const HeroBanner = () => {
  return (
    <div id="carouselExample" className="carousel slide hero-carousel">
      <div className="carousel-inner">
        <div className="carousel-item active">
          <img
            src={heroBannerImage1}
            className="d-block w-100"
            alt="Diagnóstico de Cáncer 1"
          />
        </div>
        <div className="carousel-item">
          <img
            src={heroBannerImage2}
            className="d-block w-100"
            alt="Diagnóstico de Cáncer 2"
          />
        </div>
        <div className="carousel-item">
          <img
            src={heroBannerImage3}
            className="d-block w-100"
            alt="Diagnóstico de Cáncer 3"
          />
        </div>
      </div>
      <button
        className="carousel-control-prev"
        type="button"
        data-bs-target="#carouselExample"
        data-bs-slide="prev">
        <span className="carousel-control-prev-icon" aria-hidden="true"></span>
        <span className="visually-hidden">Anterior</span>
      </button>
      <button
        className="carousel-control-next"
        type="button"
        data-bs-target="#carouselExample"
        data-bs-slide="next">
        <span className="carousel-control-next-icon" aria-hidden="true"></span>
        <span className="visually-hidden">Siguiente</span>
      </button>
    </div>
  );
};

export default HeroBanner;
