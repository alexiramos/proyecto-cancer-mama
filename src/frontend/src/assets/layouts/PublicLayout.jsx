// src/assets/layouts/PublicLayout.jsx
import React from "react";
import { Outlet } from "react-router-dom";
import Navigation from "../../components/public/Navigation";
import Footer from "../../components/public/Footer";

const PublicLayout = () => {
  return (
    <>
      <Navigation />
      <main>
        <Outlet />
      </main>
      <Footer />
    </>
  );
};

export default PublicLayout;
