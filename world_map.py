import tkinter as tk

class WorldMapRenderer:
    
    def __init__(self, canvas):
        self.canvas = canvas
    
    def project_coordinates(self, lon, lat):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        x = (lon + 180) * (canvas_width / 360)
        y = (90 - lat) * (canvas_height / 180)
        return x, y
    
    def draw_world_map(self):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        self.canvas.create_rectangle(0, 0, canvas_width, canvas_height, fill="#E6F3FF", outline="")
        
        self._draw_north_america()
        self._draw_south_america()
        self._draw_europe()
        self._draw_africa()
        self._draw_asia()
        self._draw_australia()
        self._draw_islands()
        self._draw_grid_and_labels(canvas_width, canvas_height)
    
    def _draw_north_america(self):
        north_america = [
            60, 140, 80, 120, 100, 100, 140, 80, 180, 70, 220, 80, 260, 100, 280, 130, 
            270, 160, 250, 180, 220, 190, 180, 200, 140, 210, 100, 200, 80, 180, 60, 160
        ]
        self.canvas.create_polygon(north_america, fill="#90EE90", outline="#228B22", width=2)
        
        alaska = [40, 80, 60, 60, 80, 50, 100, 60, 80, 80, 60, 90, 40, 90]
        self.canvas.create_polygon(alaska, fill="#90EE90", outline="#228B22", width=1)
        
        greenland = [180, 60, 200, 50, 220, 60, 240, 80, 220, 100, 200, 90, 180, 80]
        self.canvas.create_polygon(greenland, fill="#90EE90", outline="#228B22", width=1)
    
    def _draw_south_america(self):
        south_america = [
            140, 200, 160, 220, 180, 250, 200, 280, 190, 320, 170, 350, 150, 380, 
            130, 400, 110, 420, 90, 440, 70, 420, 90, 380, 110, 350, 130, 320, 150, 280, 170, 250
        ]
        self.canvas.create_polygon(south_america, fill="#90EE90", outline="#228B22", width=2)
    
    def _draw_europe(self):
        europe = [420, 120, 440, 100, 460, 110, 480, 130, 500, 150, 480, 170, 460, 160, 440, 150, 420, 140]
        self.canvas.create_polygon(europe, fill="#90EE90", outline="#228B22", width=2)
        
        britain = [400, 120, 420, 110, 440, 120, 420, 140, 400, 130]
        self.canvas.create_polygon(britain, fill="#90EE90", outline="#228B22", width=1)
        
        iceland = [380, 100, 400, 90, 420, 100, 400, 120, 380, 110]
        self.canvas.create_polygon(iceland, fill="#90EE90", outline="#228B22", width=1)
    
    def _draw_africa(self):
        africa = [
            440, 150, 460, 170, 480, 200, 500, 230, 480, 260, 460, 290, 440, 320, 
            420, 350, 400, 380, 380, 400, 360, 380, 380, 350, 400, 320, 420, 290, 
            440, 260, 460, 230, 480, 200, 500, 170
        ]
        self.canvas.create_polygon(africa, fill="#90EE90", outline="#228B22", width=2)
        
        madagascar = [520, 280, 540, 270, 560, 280, 540, 300, 520, 290]
        self.canvas.create_polygon(madagascar, fill="#90EE90", outline="#228B22", width=1)
    
    def _draw_asia(self):
        asia = [
            500, 150, 540, 130, 580, 120, 620, 130, 660, 150, 700, 170, 740, 190, 
            760, 220, 740, 250, 700, 270, 660, 280, 620, 270, 580, 260, 540, 240, 500, 220, 480, 200, 500, 180
        ]
        self.canvas.create_polygon(asia, fill="#90EE90", outline="#228B22", width=2)
        
        japan = [720, 150, 740, 140, 760, 150, 780, 170, 760, 190, 740, 180, 720, 170]
        self.canvas.create_polygon(japan, fill="#90EE90", outline="#228B22", width=1)
    
    def _draw_australia(self):
        australia = [640, 320, 680, 300, 720, 310, 740, 330, 720, 350, 680, 360, 640, 370, 620, 350, 640, 330]
        self.canvas.create_polygon(australia, fill="#90EE90", outline="#228B22", width=2)
        
        nz = [680, 380, 700, 370, 720, 380, 700, 400, 680, 390]
        self.canvas.create_polygon(nz, fill="#90EE90", outline="#228B22", width=1)
    
    def _draw_islands(self):
        pass
    
    def _draw_grid_and_labels(self, canvas_width, canvas_height):
        for i in range(0, canvas_width, 50):
            self.canvas.create_line(i, 0, i, canvas_height, fill="#CCCCCC", width=1, dash=(2, 2))
        for i in range(0, canvas_height, 50):
            self.canvas.create_line(0, i, canvas_width, i, fill="#CCCCCC", width=1, dash=(2, 2))
        
        self.canvas.create_text(20, 20, text="90°N", fill="#666666", font=("Arial", 8))
        self.canvas.create_text(20, canvas_height//2, text="0°", fill="#666666", font=("Arial", 8))
        self.canvas.create_text(20, canvas_height-20, text="90°S", fill="#666666", font=("Arial", 8))
        self.canvas.create_text(canvas_width//2, 20, text="180°W", fill="#666666", font=("Arial", 8))
        self.canvas.create_text(canvas_width-20, 20, text="180°E", fill="#666666", font=("Arial", 8))
        
        self.canvas.create_text(150, 150, text="North America", fill="#228B22", font=("Arial", 10, "bold"))
        self.canvas.create_text(120, 300, text="South America", fill="#228B22", font=("Arial", 10, "bold"))
        self.canvas.create_text(450, 130, text="Europe", fill="#228B22", font=("Arial", 10, "bold"))
        self.canvas.create_text(450, 250, text="Africa", fill="#228B22", font=("Arial", 10, "bold"))
        self.canvas.create_text(600, 200, text="Asia", fill="#228B22", font=("Arial", 10, "bold"))
        self.canvas.create_text(680, 340, text="Australia", fill="#228B22", font=("Arial", 10, "bold"))
    
    def draw_flight_route(self, route, airports_df):
        coords = []
        for code in route:
            airport = airports_df[airports_df["iata"] == code].iloc[0]
            x, y = self.project_coordinates(airport["longitude"], airport["latitude"])
            coords.append((x, y))
        
        if len(coords) > 1:
            self.canvas.create_line(coords, fill="#FF4444", width=4, smooth=True)
            self.canvas.create_line(coords, fill="#FF6666", width=2, smooth=True)
            
            for i in range(len(coords) - 1):
                x1, y1 = coords[i]
                x2, y2 = coords[i + 1]
                arrow_x = x1 + (x2 - x1) * 0.75
                arrow_y = y1 + (y2 - y1) * 0.75
                self.canvas.create_polygon(
                    arrow_x - 5, arrow_y - 3,
                    arrow_x + 5, arrow_y,
                    arrow_x - 5, arrow_y + 3,
                    fill="#FF0000", outline="#CC0000"
                )
        
        for i, (x, y) in enumerate(coords):
            if i == 0:
                color = "#00AA00"
                outline = "#008800"
                size = 8
            elif i == len(coords) - 1:
                color = "#FF0000"
                outline = "#CC0000"
                size = 8
            else:
                color = "#0066CC"
                outline = "#004499"
                size = 6
            
            self.canvas.create_oval(x - size - 1, y - size - 1, x + size + 1, y + size + 1, 
                                  fill="#666666", outline="")
            self.canvas.create_oval(x - size, y - size, x + size, y + size, 
                                  fill=color, outline=outline, width=2)
            
            if i == 0:
                text_anchor = tk.SE
                text_x, text_y = x - 5, y - 5
            elif i == len(coords) - 1:
                text_anchor = tk.SW
                text_x, text_y = x + 5, y - 5
            else:
                text_anchor = tk.N
                text_x, text_y = x, y - size - 15
            
            self.canvas.create_text(text_x, text_y, text=route[i], 
                                  anchor=text_anchor, font=("Arial", 9, "bold"), 
                                  fill="#333333")
            
            airport_info = airports_df[airports_df["iata"] == route[i]].iloc[0]
            city_name = airport_info["city"]
            self.canvas.create_text(text_x, text_y + 12, text=city_name, 
                                  anchor=text_anchor, font=("Arial", 7), 
                                  fill="#666666") 
