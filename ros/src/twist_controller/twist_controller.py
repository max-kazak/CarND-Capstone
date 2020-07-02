
GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, *args, **kwargs):
        # TODO: Implement
        
        #create yaw controller object
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)
        
        #create throttle PID object
        kp = 0.3
        kd = 0.
        ki = 0.1
        mn = 0.   # Minimum throttle value
        mx = 0.2  # Maximum throttle value
        self.throttle_controller = PID(kp, ki, kd, mn, mx)
        
        #create velocity low pass object
        tau = 0.5
        ts = 0.02  # sample time
        self.vel_lpf = LowPassFilter(tau, ts)
        #self.error_lpf = LowPassFilter(tau_err, self.ts)
        
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        #self.wheel_base = wheel_base
        self.wheel_radius = wheel_radius
        #self.steer_ratio = steer_ratio
        #self.min_speed = min_speed
        #self.max_lat_accel = max_lat_accel
        #self.max_steer_angle = max_steer_angle
        #self.ts = 1.0/rate
        self.accel_limit = accel_limit
        self.decel_limit = decel_limit
        
        self.last_time = rospy.get_time()
        
       
    def control(self, *args, **kwargs):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0., 0., 0.
        
        current_vel = self.vel_lpf.filt(current_vel)
       
        #filtered_error = self.error_lpf.filt(linear_vel_error)
        
        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)
        
        vel_error = linear_vel - current_vel
        self.last_vel = current_vel
        
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time
        
        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0
        
        if linear_vel == 0. and current_vel < 0.1:
            throttle = 0
            brake = 400
       # elif filtered_error < -0.08:
        #    throttle = 0.0
         #   decel = max(linear_vel_error, self.decel_limit)
          #  brake = abs(decel) * self.vehicle_mass * self.wheel_radius   
        elif throttle < 0.1 and vel_error < 0:
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius
        
        return throttle, brake, steering
