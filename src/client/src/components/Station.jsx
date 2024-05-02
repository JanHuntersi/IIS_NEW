import React, { useEffect, useState } from "react";

function Station({ station, onClose }) {
	const [predictions, setPredictions] = useState([]);
	const [time, setTime] = useState([]);
	const [isLoading, setIsLoading] = useState(true);

	useEffect(() => {
		setIsLoading(true);
		setPredictions([]);
		const givenTime = new Date();
		for (let i = 1; i < 9; i++) {
			givenTime.setMinutes(Math.round(givenTime.getMinutes() / 60) * 60);
			const roundedTime = givenTime.toLocaleTimeString("en-US", {
				hour: "2-digit",
				minute: "2-digit",
				hour12: false,
			});
			givenTime.setHours(givenTime.getHours() + 1);
			setTime((time) => [...time, roundedTime]);
		}

		const getPredictions = async () => {
			const response = await fetch(
				`http://127.0.0.1:5000/mbajk/predict/station/${station.number}`
			);
			const data = await response.json();
			console.log(data);
			setPredictions(data.predictions);
			setIsLoading(false); // Set loading state to false once predictions are fetched
		};
		getPredictions();
	}, [station]);

	return (
		<div
			style={{
				width: "300px",
				background: "white",
				boxShadow: "0 0 10px rgba(0, 0, 0, 0.1)",
				padding: "20px",
			}}
		>
			<h2>{station.name}</h2>
			<h3>Station number {station.number}</h3>
			<p>Prostih kolesarkih stojal {station.available_bike_stands}</p>
			{isLoading ? (
				<p>Loading predictions...</p>
			) : (
				<div>
					<p>Napovedi:</p>
					<ul>
						{predictions.map((prediction, index) => (
							<li key={index}>
								{time[index]} :{Math.round(prediction)}
							</li>
						))}
					</ul>
				</div>
			)}
			<button onClick={onClose}>Close</button>
		</div>
	);
}

export default Station;
