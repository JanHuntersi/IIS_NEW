import React, { useEffect, useRef } from "react";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import axios from "axios";

export default function Map() {
	const mapRef = useRef(null);
	const [stations, setStations] = React.useState([]);

	useEffect(() => {
		// Check if map is already initialized
		axios.get("http://127.0.0.1:5000/mbajk/stations").then((response) => {
			setStations(response.data);
			if (mapRef.current) {
				response.data.forEach((station) => {
					// Create a popup content with a link
					const popupContent = `<b>${station.name}</b><br /><a href="${station.link}" target="_blank">Open Link</a>`;

					// Create marker with popup and add to map
					L.marker([station.position.lat, station.position.lng])
						.addTo(mapRef.current)
						.bindPopup(popupContent);
				});
			}
		});

		if (!mapRef.current) {
			const mapInstance = L.map("map").setView([46.5547, 15.6459], 12);
			// Add tile layer (using OpenStreetMap tiles)
			L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
				attribution: "© OpenStreetMap contributors",
			}).addTo(mapInstance);
			mapRef.current = mapInstance;
		}
	}, []);

	return <div id="map" style={{ height: "90%" }}></div>;
}
