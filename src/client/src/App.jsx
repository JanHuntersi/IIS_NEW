import { useState } from "react";
import "./App.css";
import Header from "./components/Header";
import Map from "./components/Map";
import Station from "./components/Station";
import { Routes, Route } from "react-router-dom";

function App() {
	return (
		<>
			<div className="app">
				<Header />
				<Map />
			</div>
		</>
	);
}

export default App;
