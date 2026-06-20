import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import pyvisa


# =========================
# SCPI BACKEND USING PYVISA
# =========================

class SCPIPowerSupply:
    def __init__(self, resource="ASRL6::INSTR"):
        self.resource_name = resource
        self.rm = pyvisa.ResourceManager()
        self.inst = None
        self.lock = threading.Lock()

    def set_resource(self, resource):
        self.resource_name = resource

    def is_connected(self):
        with self.lock:
            return self.inst is not None

    def _set_disconnected(self):
        with self.lock:
            self.inst = None

    def connect(self):
        try:
            inst = self.rm.open_resource(self.resource_name)
            inst.timeout = 5000
            with self.lock:
                self.inst = inst
        except Exception as e:
            with self.lock:
                self.inst = None
            raise RuntimeError(f"Could not open {self.resource_name}: {e}")

    def disconnect(self):
        with self.lock:
            if self.inst:
                try:
                    self.inst.close()
                except:
                    pass
                self.inst = None

    def query(self, cmd):
        with self.lock:
            if self.inst is None:
                raise RuntimeError("Not connected to instrument")
            resp = self.inst.query(cmd).strip()

        if resp.startswith("ERR"):
            raise RuntimeError(resp)

        return resp

    def write(self, cmd):
        with self.lock:
            if self.inst is None:
                raise RuntimeError("Not connected to instrument")
            self.inst.write(cmd)

    # Convenience wrappers
    def idn(self):
        return self.query("*IDN?")

    def reset(self):
        self.write("*RST")

    def output_on(self):
        self.write("OUTP ON")

    def output_off(self):
        self.write("OUTP OFF")

    def set_voltage(self, v):
        self.write(f"SOUR:VOLT {v}")

    def set_current(self, i):
        self.write(f"SOUR:CURR {i}")

    def set_vprot(self, v):
        self.write(f"SOUR:VOLT:PROT {v}")

    def set_iprot(self, i):
        self.write(f"SOUR:CURR:PROT {i}")

    def set_mode_voltage(self):
        self.write("FUNC VOLT")

    def set_mode_current(self):
        self.write("FUNC CURR")

    def meas_voltage(self):
        resp = self.query("MEAS:VOLT?")
        return float(resp)

    def meas_current(self):
        resp = self.query("MEAS:CURR?")
        return float(resp)

    def read_both(self):
        resp = self.query("READ?")

        try:
            v_str, i_str = resp.split(",")

            v = float(v_str.strip())
            i = float(i_str.strip())

            return v, i

        except Exception:
            raise RuntimeError(f"Invalid READ? response: {resp}")

    def status(self):
        self.write("STAT?")


# =========================
# TKINTER FRONT PANEL
# =========================

class PSUApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SCPI Power Supply (PyVISA)")
        self.resizable(False, False)

        self.psu = SCPIPowerSupply("ASRL6::INSTR")
        self.output_state = False
        self.mode = tk.StringVar(value="VOLT")

        self._build_ui()
        self._start_update_thread()

    def _build_ui(self):
        main = ttk.Frame(self, padding=10)
        main.grid(row=0, column=0)

        # Connection
        conn = ttk.LabelFrame(main, text="Connection")
        conn.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        ttk.Label(conn, text="VISA Resource:").grid(row=0, column=0)
        self.resource_entry = ttk.Entry(conn, width=20)
        self.resource_entry.insert(0, "ASRL6::INSTR")
        self.resource_entry.grid(row=0, column=1, padx=5)

        ttk.Button(conn, text="Connect", command=self.on_connect).grid(row=0, column=2, padx=5)

        self.idn_label = ttk.Label(conn, text="IDN: -", width=50)
        self.idn_label.grid(row=1, column=0, columnspan=3, sticky="w", pady=5)

        # Display
        disp = ttk.LabelFrame(main, text="Display")
        disp.grid(row=1, column=0, sticky="nsew", padx=(0, 10))

        self.volt_var = tk.StringVar(value="0.000 V")
        self.curr_var = tk.StringVar(value="0.000 mA")

        volt_label = tk.Label(disp, textvariable=self.volt_var,
                              font=("Consolas", 24), fg="lime", bg="black", width=10)
        volt_label.grid(row=0, column=0, padx=5, pady=5)

        curr_label = tk.Label(disp, textvariable=self.curr_var,
                              font=("Consolas", 24), fg="cyan", bg="black", width=10)
        curr_label.grid(row=1, column=0, padx=5, pady=5)

        # Controls
        ctrl = ttk.LabelFrame(main, text="Controls")
        ctrl.grid(row=1, column=1, sticky="nsew")

        ttk.Label(ctrl, text="Voltage (V):").grid(row=0, column=0)
        self.vset = ttk.Entry(ctrl, width=8)
        self.vset.insert(0, "0.00")
        self.vset.grid(row=0, column=1, padx=5)
        ttk.Button(ctrl, text="Apply", command=self.on_set_voltage).grid(row=0, column=2)

        ttk.Label(ctrl, text="Current (mA):").grid(row=1, column=0)
        self.iset = ttk.Entry(ctrl, width=8)
        self.iset.insert(0, "0.00")
        self.iset.grid(row=1, column=1, padx=5)
        ttk.Button(ctrl, text="Apply", command=self.on_set_current).grid(row=1, column=2)

        ttk.Label(ctrl, text="OVP (V):").grid(row=2, column=0)
        self.ovp = ttk.Entry(ctrl, width=8)
        self.ovp.insert(0, "5.00")
        self.ovp.grid(row=2, column=1, padx=5)
        ttk.Button(ctrl, text="Set", command=self.on_set_ovp).grid(row=2, column=2)

        ttk.Label(ctrl, text="OCP (mA):").grid(row=3, column=0)
        self.ocp = ttk.Entry(ctrl, width=8)
        self.ocp.insert(0, "250")
        self.ocp.grid(row=3, column=1, padx=5)
        ttk.Button(ctrl, text="Set", command=self.on_set_ocp).grid(row=3, column=2)

        # Mode
        mode_frame = ttk.LabelFrame(ctrl, text="Mode")
        mode_frame.grid(row=4, column=0, columnspan=3, pady=10)

        ttk.Radiobutton(mode_frame, text="Voltage", variable=self.mode, value="VOLT",
                        command=self.on_mode_change).grid(row=0, column=0, padx=5)
        ttk.Radiobutton(mode_frame, text="Current", variable=self.mode, value="CURR",
                        command=self.on_mode_change).grid(row=0, column=1, padx=5)

        # Output
        out = ttk.LabelFrame(ctrl, text="Output")
        out.grid(row=5, column=0, columnspan=3, pady=10)

        self.out_btn = ttk.Button(out, text="Power ON/OFF", command=self.on_toggle_output)
        self.out_btn.grid(row=0, column=0, padx=5)

        self.led = tk.Canvas(out, width=25, height=25)
        self.led.grid(row=0, column=1)
        self.led_id = self.led.create_oval(2, 2, 25, 25, fill="red")

        # Status
        bottom = ttk.Frame(main)
        bottom.grid(row=2, column=0, columnspan=2, pady=10)

        ttk.Button(bottom, text="*RST", command=self.on_reset).grid(row=0, column=0, padx=5)
        ttk.Button(bottom, text="STAT?", command=self.on_status).grid(row=0, column=1, padx=5)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(bottom, textvariable=self.status_var, width=60, anchor="w").grid(row=0, column=2)

        # ---------------------------------------------------------
        # SCPI Console
        # ---------------------------------------------------------
        console_frame = ttk.LabelFrame(main, text="SCPI Console")
        console_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(10, 0))

        # Text output area
        self.console_text = tk.Text(console_frame, height=10, width=70,
                                    bg="black", fg="lime", insertbackground="white")
        self.console_text.grid(row=0, column=0, columnspan=2, padx=5, pady=5)

        # Scrollbar
        scroll = ttk.Scrollbar(console_frame, command=self.console_text.yview)
        scroll.grid(row=0, column=2, sticky="ns")
        self.console_text.config(yscrollcommand=scroll.set)

        # Entry box
        self.console_entry = ttk.Entry(console_frame, width=60)
        self.console_entry.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.console_entry.bind("<Return>", self.on_console_enter)

        # Send button
        ttk.Button(console_frame, text="Send", command=self.on_console_send).grid(
            row=1, column=1, padx=5, pady=5, sticky="e"
        )
        self.bind_scroll(self.vset, step=0.1, min_val=0)
        self.bind_scroll(self.iset, step=1.0, min_val=0)
        self.bind_scroll(self.ovp, step=0.1, min_val=0)
        self.bind_scroll(self.ocp, step=1.0, min_val=0)

        # Voltage
        self.vset.bind("<Return>", lambda e: self.on_set_voltage())

        # Current
        self.iset.bind("<Return>", lambda e: self.on_set_current())

        # OVP
        self.ovp.bind("<Return>", lambda e: self.on_set_ovp())

        # OCP
        self.ocp.bind("<Return>", lambda e: self.on_set_ocp())

    # =========================
    # CALLBACKS
    # =========================

    def on_connect(self):
        resource = self.resource_entry.get().strip()
        self.psu.set_resource(resource)
        try:
            self.psu.connect()
            idn = self.psu.idn()
            self.idn_label.config(text=f"IDN: {idn}")
            self.status_var.set(f"Connected to {resource}")
        except Exception as e:
            messagebox.showerror("Connection error", str(e))

    def on_set_voltage(self):
        try:
            self.psu.set_voltage(float(self.vset.get()))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_set_current(self):
        try:
            self.psu.set_current(float(self.iset.get()))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_set_ovp(self):
        try:
            self.psu.set_vprot(float(self.ovp.get()))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_set_ocp(self):
        try:
            self.psu.set_iprot(float(self.ocp.get()))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_mode_change(self):
        try:
            if self.mode.get() == "VOLT":
                self.psu.set_mode_voltage()
            else:
                self.psu.set_mode_current()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_toggle_output(self):
        try:
            if self.output_state:
                self.psu.output_off()
                self.output_state = False
                self.out_btn.config(text="Power ON/OFF")
                self.led.itemconfig(self.led_id, fill="grey")
            else:
                self.psu.output_on()
                self.output_state = True
                self.out_btn.config(text="Power ON/OFF")
                self.led.itemconfig(self.led_id, fill="red")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_reset(self):
        try:
            self.psu.reset()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_status(self):
        try:
            self.psu.status()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # =========================
    # UPDATE THREAD
    # =========================

    def _start_update_thread(self):
        threading.Thread(target=self._update_loop, daemon=True).start()

    def _update_loop(self):
        last_v, last_i = 0.0, 0.0  # keep last known good readings
        while True:
            if not self.psu.is_connected():
                time.sleep(0.2)
                continue

            try:
                try:
                    v, i = self.psu.read_both()
                    last_v, last_i = v, i
                except Exception as e:
                    # fallback: try individual measurements
                    try:
                        v = self.psu.meas_voltage()
                        i = self.psu.meas_current()
                        last_v, last_i = v, i
                    except:
                        # use last known values
                        v, i = last_v, last_i

                # Poll output state and mode safely
                try:
                    out_state = self.psu.query("OUTP?").strip().upper()
                    if out_state not in ("ON", "OFF"):
                        raise ValueError("Invalid OUTP response")
                except:
                    out_state = "ON" if self.output_state else "OFF"

            
                try:
                    mode_state = self.psu.query("FUNC?").strip().upper()
                    if mode_state not in ("VOLT", "CURR"):
                        raise ValueError("Invalid FUNC response")
                except:
                    mode_state = self.mode.get()

                # Update GUI safely
                self.after(0, self._update_display, v, i)
                self.after(0, self._update_state, out_state, mode_state)

            except Exception as e:
                print("UPDATE LOOP ERROR:", e)
                if "VI_ERROR" in str(e) or "Not connected" in str(e):
                    self.psu._set_disconnected()
                    self.after(0, lambda: self.status_var.set("Disconnected"))

            time.sleep(0.2)

    def _update_display(self, v, i):
        self.volt_var.set(f"{v:7.3f} V")
        self.curr_var.set(f"{i:7.3f} mA")

    def _update_state(self, out_state, mode_state):

        # --- OUTPUT ---
        if out_state == "ON":
            self.output_state = True
            self.led.itemconfig(self.led_id, fill="red")
        elif out_state == "OFF":
            self.output_state = False
            self.led.itemconfig(self.led_id, fill="grey")
        # else: ignore invalid

        # --- MODE ---
        if mode_state in ("VOLT", "CURR"):
            self.mode.set(mode_state)


    def console_write(self, text):
        """Append text to the console with auto-scroll."""
        self.console_text.insert(tk.END, text + "\n")
        self.console_text.see(tk.END)

    def on_console_enter(self, event):
        self.on_console_send()

    def _console_worker(self, cmd):
        try:
            if cmd.endswith("?"):
                resp = self.psu.query(cmd)
            else:
                self.psu.write(cmd)
                resp = "OK"

                # # --- update GUI for output/mode changes ---
                # if cmd.upper() in ("OUTP ON", "OUTP OFF"):
                #     out_state = "ON" if "ON" in cmd.upper() else "OFF"
                #     self.after(0, lambda: self._update_state(out_state, self.mode.get()))
                # elif cmd.upper() in ("FUNC VOLT", "FUNC CURR"):
                #     self.after(0, lambda: self._update_state(
                #         self.out_btn.cget("text").split()[-1], cmd.split()[-1]
                #     ))

        except Exception as e:
            resp = f"ERR: {e}"
            self.psu._set_disconnected()
            self.after(0, lambda: self.status_var.set("Disconnected"))

        self.after(0, lambda: self.console_write(resp))

    def on_console_send(self):
        cmd = self.console_entry.get().strip()
        if not cmd:
            return

        self.console_entry.delete(0, tk.END)
        self.console_write(f"> {cmd}")

        if not self.psu.is_connected():
            self.console_write("! Not connected")
            return

        # ✅ run in background thread
        threading.Thread(target=self._console_worker, args=(cmd,), daemon=True).start()
    
    def bind_scroll(self, widget, step=0.1, min_val=None, max_val=None):
        def on_scroll(event):
            try:
                value = float(widget.get())
            except ValueError:
                value = 0.0

            # Windows / Linux
            if event.delta:
                delta = step if event.delta > 0 else -step
            else:
                # Linux (event.num)
                delta = step if event.num == 4 else -step

            value += delta

            if min_val is not None:
                value = max(min_val, value)
            if max_val is not None:
                value = min(max_val, value)

            widget.delete(0, tk.END)
            widget.insert(0, f"{value:.3f}")

        # Windows / macOS
        widget.bind("<MouseWheel>", on_scroll)

        # Linux support
        widget.bind("<Button-4>", on_scroll)
        widget.bind("<Button-5>", on_scroll)




if __name__ == "__main__":
    app = PSUApp()
    app.mainloop()