# Option Pricing & Volatility Surface Simulator

Interactive Python simulator for option pricing, Black-Scholes valuation, spot-path simulation, Greeks monitoring, moneyness tracking, and animated 3D price and implied-volatility surfaces.

## Overview

This project is a quantitative finance simulator built in Python to visualize the dynamics of a European call option under the Black-Scholes framework.

The application simulates a stochastic spot path, computes the corresponding option price and Greeks over time, and displays several interactive 3D views, including dynamic price surfaces, tangent surfaces and implied-volatility surfaces.

The goal is to connect option-pricing theory with visual and interactive market-style representations.

## Features

* Black-Scholes valuation for European call options
* Simulated stochastic spot path
* Dynamic option price evolution
* Moneyness tracking: ITM, ATM, OTM, deep ITM and deep OTM
* Core Greeks:

  * Delta
  * Gamma
  * Vega
  * Theta
* Advanced Greeks:

  * Vanna
  * Vomma
  * Charm
  * Color
  * Speed
  * Zomma
* Animated 3D option price surfaces
* Implied-volatility surface simulation
* Interactive controls:

  * Play / pause
  * New generation
  * Finish animation
  * Display / hide selected surfaces
  * Transparency sliders
  * Input fields for spot, strike and maturity

## Tech Stack

* Python
* NumPy
* Matplotlib
* Matplotlib 3D / mplot3d
* Matplotlib widgets
* Matplotlib animation

## Installation

Clone the repository:

```bash
git clone https://github.com/LucThiebaut/option-pricing-volatility-surface.git
cd option-pricing-volatility-surface
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Requirements

The project requires:

```txt
numpy
matplotlib
```

## Usage

Run the main Python file:

```bash
python option_pricing_vol_surface.py
```

The simulator will open several interactive Matplotlib windows displaying the option path, pricing surfaces and volatility surfaces.

## Project Structure

```text
option-pricing-volatility-surface/
│
├── option_pricing_vol_surface.py
├── requirements.txt
└── README.md
```

## Financial Concepts Covered

This project illustrates several key concepts in quantitative finance:

* Risk-neutral option pricing
* Black-Scholes model
* Spot path simulation
* Time-to-maturity dynamics
* Option sensitivity analysis through Greeks
* Moneyness and payoff behavior
* Volatility surface visualization
* 3D representation of option-pricing relationships

## Educational Purpose

This project was developed as a personal quantitative finance project to strengthen the link between mathematical finance, programming and market-oriented option-pricing intuition.

It is intended for educational and demonstrative purposes only and should not be used as a production trading or risk-management system.

## Author

Luc Thiebaut
LinkedIn: [www.linkedin.com/in/lucthiebaut/](http://www.linkedin.com/in/lucthiebaut/)
