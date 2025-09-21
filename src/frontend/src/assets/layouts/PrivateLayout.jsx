// src/assets/layouts/PrivateLayout.jsx
import React from "react";
import { Outlet } from "react-router-dom";
import Sidebar from "../../components/private/Sidebar";

const PrivateLayout = () => {
  return (
    <div className="d-flex">
      <Sidebar />
      <main className="flex-grow-1 p-4">
        <Outlet />
      </main>
    </div>
  );
};

export default PrivateLayout;
