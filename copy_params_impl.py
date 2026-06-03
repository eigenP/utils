def _copy_params(self, change):
    s = []

    # physical pos/thickness
    x_s_phys = self.x_s * self.sx
    y_s_phys = self.y_s * self.sy
    z_s_phys = self.z_s * self.sz
    s.append(f"slabs_position=({z_s_phys:.2f}, {y_s_phys:.2f}, {x_s_phys:.2f}),")

    x_t_phys = self.x_t * self.sx
    y_t_phys = self.y_t * self.sy
    z_t_phys = self.z_t * self.sz
    s.append(f"slabs_thickness=({z_t_phys:.2f}, {y_t_phys:.2f}, {x_t_phys:.2f}),")

    # channels
    vmin = [None if v == "" else float(v) for v in self.vmin_list]
    vmax = [None if v == "" else float(v) for v in self.vmax_list]
    gamma = [float(g) for g in self.gamma_list]
    opacity = [float(o) for o in self.opacity_list]

    s.append(f"vmin={vmin},")
    s.append(f"vmax={vmax},")
    s.append(f"gamma={gamma},")
    s.append(f"opacity={opacity},")

    # any string rotation?
    if hasattr(self, 'rotate_view'):
        s.append(f"rotate_view={self.rotate_view},")

    print("\n".join(s))
