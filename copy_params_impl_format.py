class Mock:
    def __init__(self):
        self.x_s = 20
        self.y_s = 30
        self.z_s = 40
        self.x_t = 5
        self.y_t = 6
        self.z_t = 7
        self.sx = 0.5
        self.sy = 0.5
        self.sz = 2.0
        self.vmin_list = [0.0, 10.0]
        self.vmax_list = [255.0, 200.0]
        self.gamma_list = [1.0, 0.5]
        self.opacity_list = [1.0, 1.0]
        self.rotate_view = (10, 0, 0)
        self.C = 2

def get_params_str(self):
    s = []

    # physical pos/thickness
    x_s_phys = self.x_s * self.sx
    y_s_phys = self.y_s * self.sy
    z_s_phys = self.z_s * self.sz
    s.append(f"slabs_position=({z_s_phys}, {y_s_phys}, {x_s_phys}),")

    x_t_phys = self.x_t * self.sx
    y_t_phys = self.y_t * self.sy
    z_t_phys = self.z_t * self.sz
    s.append(f"slabs_thickness=({z_t_phys}, {y_t_phys}, {x_t_phys}),")

    # channels
    # The length depends on num channels
    if hasattr(self, 'num_channels'):
        n = self.num_channels
    else:
        n = getattr(self, 'C', 1)

    vmin = [None if v == "" else float(v) for v in self.vmin_list][:n]
    vmax = [None if v == "" else float(v) for v in self.vmax_list][:n]
    gamma = [float(g) for g in self.gamma_list][:n]
    opacity = [float(o) for o in self.opacity_list][:n]

    s.append(f"vmin={vmin},")
    s.append(f"vmax={vmax},")
    s.append(f"gamma={gamma},")
    s.append(f"opacity={opacity},")

    # any string rotation?
    if hasattr(self, 'rotate_view'):
        s.append(f"rotate_view={self.rotate_view},")

    return "\n".join(s)

m = Mock()
print(get_params_str(m))
