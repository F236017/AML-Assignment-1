# LinkedIn Post Draft (Task 6.3)

☀️ **Can free satellite weather data replace multi-thousand-dollar on-site solar sensors?**

As renewable energy scales globally, precise solar power forecasting is vital for grid stability and project financing. But while utility-scale solar farms deploy high-grade pyranometers and module thermal sensors, residential and commercial rooftop installers usually have no hardware sensors—only free public weather services.

In our latest **Applied Machine Learning (AML)** project, my teammate and I set out to quantify the exact trade-off:

👉 **"How much prediction accuracy is lost when on-site irradiation and temperature sensors are replaced by free public weather data from the internet?"**

### 🔍 What We Did:
1. **Built Everything From Scratch**: Implemented linear regression models purely using NumPy from first principles—Normal Equation, Batch Gradient Descent, and Stochastic Gradient Descent. (Zero calls to scikit-learn or statsmodels!).
2. **Real Industrial Data**: Analyzed 34 days of 15-minute telemetry from 22 central inverters at Solar Plant 1 in Gandikota, Andhra Pradesh, aggregated to 816 hourly records.
3. **Public Weather API Integration**: Queried the Open-Meteo Historical Archive API, performing geographical sanity checks by verifying zero phase lag between satellite shortwave radiation and on-site pyranometer peaks.
4. **Evaluation**: Enforced a strict chronological train/test split (27 days train / 7 days test) and evaluated daytime RMSE (irradiation > 0).

### 📊 Key Findings:
- **On-Site Sensors (Set A)**: Achieved a daytime test RMSE of **725.16 kW** — just **2.65%** of the plant's 27.3 MW peak capacity!
- **Public Weather (Set B, Open-Meteo)**: Reached a daytime test RMSE of **3,427.06 kW** (**12.54%** of peak capacity).
- **The Verdict**: Replacing dedicated pyranometers with satellite data increases error by **+9.89%** of plant capacity. While this 4.7× variance penalty is unacceptable for real-time transmission grid balancing, **it is a game-changer for rooftop solar installers**! For rooftop financial payback modeling and billing estimates, monthly errors average out to under 4%, saving thousands in sensor capital expenditures.

We also built an interactive web predictor to serve real-time inferences from our trained weights!

📖 **Read our full deep-dive on Medium**: [Link to Medium Article]  
💻 **Explore the Code on GitHub**: [Link to GitHub Repository]

#MachineLearning #SolarEnergy #RenewableEnergy #DataScience #Python #NumPy #CleanTech #ArtificialIntelligence #EnergyForecasting
