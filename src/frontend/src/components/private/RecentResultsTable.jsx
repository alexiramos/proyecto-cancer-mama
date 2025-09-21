// src/components/private/RecentResultsTable.jsx
import React from "react";

const RecentResultsTable = () => {
  const results = [
    {
      date: "2024-07-26",
      type: "Mamografía",
      result: "Negativo",
      statusClass: "text-bg-light border",
    },
    {
      date: "2024-07-20",
      type: "Mamografía",
      result: "Positivo",
      statusClass: "text-bg-light border",
    },
    {
      date: "2024-07-15",
      type: "Mamografía",
      result: "Inconcluso",
      statusClass: "text-bg-light border",
    },
  ];

  return (
    <div className="card shadow-sm border-0">
      <div className="card-body p-4">
        <h2 className="card-title h4 mb-4">Resultados Recientes</h2>
        <div className="table-responsive">
          <table className="table table-hover align-middle">
            <thead className="table-light">
              <tr>
                <th scope="col" className="py-3">
                  Fecha
                </th>
                <th scope="col" className="py-3">
                  Tipo de Imagen
                </th>
                <th scope="col" className="py-3">
                  Resultado
                </th>
                <th scope="col" className="py-3">
                  Acciones
                </th>
              </tr>
            </thead>
            <tbody>
              {results.map((item, index) => (
                <tr key={index}>
                  <td className="py-3">{item.date}</td>
                  <td className="py-3">{item.type}</td>
                  <td className="py-3">
                    <span className={`badge ${item.statusClass} fw-semibold`}>
                      {item.result}
                    </span>
                  </td>
                  <td className="py-3">
                    <a
                      href="#"
                      className="btn btn-sm btn-link text-decoration-none fw-semibold">
                      Ver Detalles
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default RecentResultsTable;
