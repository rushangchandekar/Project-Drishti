"""
detection/density_calculator.py
Intelligent density calculation with calibration support
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DensityMetrics:
    """Density calculation results"""
    person_count: int
    density_value: float
    density_unit: str
    level: str  # EMPTY, VERY_LOW, LOW, MODERATE, HIGH, VERY_HIGH, CRITICAL
    is_calibrated: bool
    area_m2: Optional[float]
    risk_factor: float  # 0.0 to 1.0


class IntelligentDensityCalculator:
    """
    Calculate crowd density with multiple modes:
    - Calibrated (accurate people/m²)
    - Uncalibrated (adaptive thresholds)
    - Event-specific (configurable for venue type)
    """
    
    def __init__(self, mode='uncalibrated', venue_area_m2=None, venue_type='general'):
        """
        Args:
            mode: 'calibrated' or 'uncalibrated'
            venue_area_m2: Total monitored area in square meters
            venue_type: 'stadium', 'concert', 'religious', 'general'
        """
        self.mode = mode
        self.venue_area_m2 = venue_area_m2
        self.venue_type = venue_type
        
        # Safety thresholds (people/m²) based on international standards
        self.safety_thresholds = {
            'stadium': {
                'very_low': 0.5,
                'low': 1.0,
                'moderate': 2.0,
                'high': 3.0,
                'very_high': 4.0,
                'critical': 5.0
            },
            'concert': {
                'very_low': 0.3,
                'low': 0.8,
                'moderate': 1.5,
                'high': 2.5,
                'very_high': 3.5,
                'critical': 4.5
            },
            'religious': {
                'very_low': 0.4,
                'low': 1.0,
                'moderate': 2.0,
                'high': 3.0,
                'very_high': 4.5,
                'critical': 6.0
            },
            'general': {
                'very_low': 0.5,
                'low': 1.0,
                'moderate': 2.0,
                'high': 3.0,
                'very_high': 4.0,
                'critical': 5.0
            }
        }
        
        # Adaptive thresholds for uncalibrated mode
        self.count_thresholds = {
            'very_low': 5,
            'low': 15,
            'moderate': 30,
            'high': 50,
            'very_high': 80,
            'critical': 120
        }
        
        print(f"[DENSITY CALCULATOR] Mode: {mode}, Venue: {venue_type}")
    
    def calculate(self, person_count: int, frame_shape: Optional[Tuple] = None) -> DensityMetrics:
        """
        Calculate density metrics
        """
        if self.mode == 'calibrated' and self.venue_area_m2:
            return self._calculate_calibrated(person_count)
        else:
            return self._calculate_uncalibrated(person_count, frame_shape)
    
    def _calculate_calibrated(self, person_count: int) -> DensityMetrics:
        """
        Calculate actual density (people/m²)
        """
        density = person_count / self.venue_area_m2
        
        # Get density level
        thresholds = self.safety_thresholds[self.venue_type]
        level = self._get_level_from_density(density, thresholds)
        
        # Calculate risk factor
        risk = self._calculate_risk_factor(density, thresholds)
        
        return DensityMetrics(
            person_count=person_count,
            density_value=round(density, 2),
            density_unit='people/m²',
            level=level,
            is_calibrated=True,
            area_m2=self.venue_area_m2,
            risk_factor=risk
        )
    
    def _calculate_uncalibrated(self, person_count: int, frame_shape: Optional[Tuple]) -> DensityMetrics:
        """
        Calculate density using adaptive thresholds
        """
        # Estimate area based on frame size (very rough)
        if frame_shape:
            h, w = frame_shape[:2]
            pixels = h * w
            # Heuristic: assume 100-300 m² coverage
            estimated_area = 100 + (pixels / 2000000) * 200
        else:
            estimated_area = 150  # Default assumption
        
        # Estimated density
        estimated_density = person_count / estimated_area
        
        # Get level based on count
        level = self._get_level_from_count(person_count)
        
        # Calculate risk
        thresholds = self.count_thresholds
        risk = self._calculate_risk_from_count(person_count, thresholds)
        
        return DensityMetrics(
            person_count=person_count,
            density_value=round(estimated_density, 2),
            density_unit='estimated people/m²',
            level=level,
            is_calibrated=False,
            area_m2=estimated_area,
            risk_factor=risk
        )
    
    def _get_level_from_density(self, density: float, thresholds: Dict) -> str:
        """Get density level from actual density value"""
        if density == 0:
            return 'EMPTY'
        elif density < thresholds['very_low']:
            return 'VERY_LOW'
        elif density < thresholds['low']:
            return 'LOW'
        elif density < thresholds['moderate']:
            return 'MODERATE'
        elif density < thresholds['high']:
            return 'HIGH'
        elif density < thresholds['very_high']:
            return 'VERY_HIGH'
        else:
            return 'CRITICAL'
    
    def _get_level_from_count(self, count: int) -> str:
        """Get density level from person count"""
        if count == 0:
            return 'EMPTY'
        elif count <= self.count_thresholds['very_low']:
            return 'VERY_LOW'
        elif count <= self.count_thresholds['low']:
            return 'LOW'
        elif count <= self.count_thresholds['moderate']:
            return 'MODERATE'
        elif count <= self.count_thresholds['high']:
            return 'HIGH'
        elif count <= self.count_thresholds['very_high']:
            return 'VERY_HIGH'
        else:
            return 'CRITICAL'
    
    def _calculate_risk_factor(self, density: float, thresholds: Dict) -> float:
        """
        Calculate risk factor (0.0 to 1.0) from density
        """
        critical_threshold = thresholds['critical']
        
        if density >= critical_threshold:
            return 1.0
        elif density == 0:
            return 0.0
        else:
            # Linear scaling
            return min(1.0, density / critical_threshold)
    
    def _calculate_risk_from_count(self, count: int, thresholds: Dict) -> float:
        """
        Calculate risk factor from count
        """
        critical_count = thresholds['critical']
        
        if count >= critical_count:
            return 1.0
        elif count == 0:
            return 0.0
        else:
            return min(1.0, count / critical_count)
    
    def calibrate(self, reference_area_m2: float):
        """
        Calibrate calculator with actual area
        """
        self.mode = 'calibrated'
        self.venue_area_m2 = reference_area_m2
        print(f"[DENSITY CALCULATOR] Calibrated for {reference_area_m2} m²")
    
    def set_custom_thresholds(self, thresholds: Dict[str, int]):
        """
        Set custom count thresholds
        """
        self.count_thresholds.update(thresholds)
        print(f"[DENSITY CALCULATOR] Updated thresholds: {self.count_thresholds}")


# Test
if __name__ == "__main__":
    # Test uncalibrated mode
    calc_uncal = IntelligentDensityCalculator(mode='uncalibrated', venue_type='concert')
    
    test_counts = [0, 5, 15, 30, 50, 80, 120, 200]
    
    print("\n=== UNCALIBRATED MODE ===")
    for count in test_counts:
        result = calc_uncal.calculate(count, frame_shape=(720, 1280))
        print(f"\nPeople: {count:3d} → Level: {result.level:12s} | "
              f"Density: {result.density_value:.2f} {result.density_unit} | "
              f"Risk: {result.risk_factor:.2f}")
    
    # Test calibrated mode
    print("\n\n=== CALIBRATED MODE (200 m² stadium) ===")
    calc_cal = IntelligentDensityCalculator(
        mode='calibrated',
        venue_area_m2=200,
        venue_type='stadium'
    )
    
    for count in test_counts:
        result = calc_cal.calculate(count)
        print(f"\nPeople: {count:3d} → Level: {result.level:12s} | "
              f"Density: {result.density_value:.2f} {result.density_unit} | "
              f"Risk: {result.risk_factor:.2f}")