import React, { useEffect, useRef, useState } from "react";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import axios from "axios";
import Station from "./Station";
import CustomMarkerIcon from "../../assets/bike.png";

export default function Map() {
	const mapRef = useRef(null);
	const [stations, setStations] = useState([]);
	const [selectedStation, setSelectedStation] = useState(null);

	useEffect(() => {
		// Check if map is already initialized
		axios.get("http://127.0.0.1:5000/mbajk/stations").then((response) => {
			setStations(response.data);
			if (mapRef.current) {
				response.data.forEach((station) => {
					// Create a popup content with a link
					const popupContent = `<b>${station.name}</b><br />`;

					// Create marker with popup and add to map
					const marker = L.marker(
						[station.position.lat, station.position.lng],
						{
							icon: L.icon({
								iconUrl: CustomMarkerIcon,
								iconSize: [32, 32], // Set the size of your custom icon
								iconAnchor: [16, 32], // Set the anchor point of your custom icon
							}),
						}
					)
						.addTo(mapRef.current)
						.bindPopup(popupContent);

					// Add click event listener to the marker
					marker.on("click", () => {
						setSelectedStation(station);
					});
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

	return (
		<div style={{ display: "flex", height: "90%" }}>
			<div id="map" style={{ flex: "1 1 auto", height: "100%" }}></div>
			{selectedStation && (
				<Station
					station={selectedStation}
					onClose={() => setSelectedStation(null)}
				/>
			)}
		</div>
	);
}
