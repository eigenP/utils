import traitlets
import anywidget

class Widget(anywidget.AnyWidget):
    copy_params_trigger = traitlets.Int(0).tag(sync=True)
    copy_params_string = traitlets.Unicode("").tag(sync=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observe(self._copy_params, names='copy_params_trigger')

    def _copy_params(self, change):
        print("copy params!")
        self.copy_params_string = "copied!"
