# Flight Route Finder

A comprehensive flight route planning and cost analysis tool with interactive world map visualization.

![Flight Route Finder](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

## Features

### Advanced Cost Analysis
- **Comprehensive Cost Breakdown**: Fuel, operating, aircraft, and airport fees
- **Range-Based Estimates**: Low, expected, and high cost projections
- **Route Classification**: Domestic, International, and Long-haul routes
- **Realistic Factors**: Seasonal variations, route efficiency, and market conditions
- **Ticket Pricing**: Per-person pricing with profit margin calculations

### Smart City Search
- **Autocomplete Functionality**: Real-time city suggestions as you type
- **Global Airport Database**: Access to thousands of airports worldwide
- **Intelligent Matching**: Partial name matching for easy city selection

### Detailed Flight Information
- **Distance & Time**: Accurate flight distance and duration calculations
- **Route Details**: Number of stops and route type classification
- **Cost Breakdown**: Detailed analysis of all operational costs
- **Price Estimates**: Realistic ticket price ranges

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/flight-route-finder.git
   cd flight-route-finder
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python main.py
   ```

## Project Structure

```
flight-route-finder/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── LICENSE                # MIT License
├── src/                   # Source code modules
│   ├── __init__.py
│   ├── data_loader.py     # Airport and route data handling
│   ├── route_finder.py    # Pathfinding algorithms
│   ├── cost_analyzer.py   # Cost calculation engine
│   └── world_map.py       # Map rendering and visualization
└── FlightPath.py          # Legacy single-file version
```

## Usage

### Basic Operation
1. **Enter Origin City**: Type the starting city name
2. **Enter Destination City**: Type the destination city name
3. **Click "Find Route"**: The application will search for optimal routes
4. **View Results**: See the route on the map and detailed cost analysis

### Features Explained

#### **City Autocomplete**
- Start typing a city name
- Select from the dropdown suggestions
- Works for both origin and destination fields

#### **Cost Analysis**
- **Fuel Cost**: Varies by route type and market conditions
- **Operating Cost**: Crew, maintenance, and operational expenses
- **Aircraft Cost**: Depreciation, insurance, and aircraft-related costs
- **Airport Fees**: Landing fees, taxes, and service charges
- **Total Cost**: Combined with seasonal and efficiency adjustments

#### **Route Types**
- **Domestic**: < 1,000 km (lower costs)
- **International**: 1,000-5,000 km (medium costs)
- **Long-haul**: > 5,000 km (higher costs)

## Technical Details

### **Algorithms**
- **Dijkstra's Algorithm**: For finding shortest flight routes
- **Geodesic Distance**: Accurate distance calculations using geopy
- **Cost Modeling**: Realistic airline cost structure simulation

### **Data Sources**
- **OpenFlights Database**: Comprehensive airport and route data
- **Real-time Loading**: Data fetched from official OpenFlights repository

### **Technologies Used**
- **Python 3.8+**: Core programming language
- **Tkinter**: GUI framework
- **Pandas**: Data manipulation and analysis
- **Geopy**: Geographic calculations
- **PIL/Pillow**: Image processing

## Cost Analysis Methodology

### **Fuel Costs**
- Domestic: $0.70-0.90 per liter
- International: $0.80-1.20 per liter
- Long-haul: $0.90-1.40 per liter

### **Operating Costs**
- Domestic: $0.15-0.25 per km
- International: $0.20-0.35 per km
- Long-haul: $0.25-0.45 per km

### **Efficiency Factors**
- **Direct Flights**: 15% cost reduction
- **Multiple Stops**: 25% cost premium
- **Seasonal Variation**: ±30% based on demand

### **Profit Margins**
- **Ticket Pricing**: 20-40% markup on total costs
- **Seat Capacity**: 150 passengers per flight

## Map Features

### **Visual Elements**
- **Grid System**: Latitude/longitude reference lines
- **Labels**: Continent names and coordinate markers

### **Flight Route Display**
- **Route Lines**: Red lines with shadow effects
- **Directional Arrows**: Show flight direction
- **Airport Markers**: Color-coded with shadows
- **Information Labels**: Airport codes and city names

## Contributing

### **Development Setup**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **OpenFlights**: For providing comprehensive airport and route data
- **Geopy**: For accurate geographic distance calculations
- **Tkinter**: For the GUI framework
- **Pandas**: For efficient data manipulation

## Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Check the existing issues for solutions
- Review the documentation above

## Future Enhancements

- [ ] Real-time flight data integration
- [ ] Multiple route options display
- [ ] Export functionality (PDF, CSV)
- [ ] Mobile-friendly web version
- [ ] Advanced filtering options
- [ ] Historical cost trends
- [ ] Weather impact analysis

---

**Made with love for aviation enthusiasts and travelers worldwide**
