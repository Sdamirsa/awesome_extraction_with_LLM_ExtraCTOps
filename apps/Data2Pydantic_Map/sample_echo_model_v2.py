from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class GenderEnum(str, Enum):
    MALE = "Male"
    FEMALE = "Female"
    OTHER = "Other"

class ASDTypeSimplified(str, Enum):
    SECUNDUM = "Secundum"
    PRIMUM = "Primum"
    PFO = "Patent Foramen Ovale"
    SINUS_VENOSUS = "Sinus Venosus"
    OTHER = "Other"

class ShuntSizeEnum(str, Enum):
    TINY = "Tiny"
    SMALL = "Small"
    MODERATE = "Moderate"
    LARGE = "Large"
    NOT_APPLICABLE = "Not Applicable"

class ValveStenosisSeverity(str, Enum):
    NO = "No"
    MILD = "Mild"
    MODERATE = "Moderate"
    SEVERE = "Severe"

class AorticValveSimplified(BaseModel):
    stenosis_severity: Optional[ValveStenosisSeverity] = Field(None, description="Severity of aortic valve stenosis.")
    peak_gradient: Optional[float] = Field(None, description="Peak pressure gradient across the aortic valve in mmHg.")
    is_bicuspid: Optional[bool] = Field(None, description="Is the aortic valve bicuspid?")

class EchoReportV2(BaseModel):
    patient_id: str = Field(description="Unique patient identifier.")
    report_id: str = Field(description="Unique report identifier.")
    age_at_echo: Optional[int] = Field(None, description="Patient's age at the time of the echo in years.")
    height_cm: Optional[float] = Field(None, description="Patient's height in centimeters.")
    is_urgent_study: Optional[bool] = Field(None, description="Was this an urgent study?")
    gender: Optional[GenderEnum] = Field(None, description="Patient's gender.")
    
    symptoms_reported: Optional[List[str]] = Field(None, description="List of reported symptoms.")
    
    asd_present: Optional[bool] = Field(None, description="Is an Atrial Septal Defect (ASD) or Patent Foramen Ovale (PFO) present?")
    asd_types: Optional[List[ASDTypeSimplified]] = Field(None, description="Types of ASD/PFO observed.")
    asd_size: Optional[ShuntSizeEnum] = Field(None, description="Size of the ASD/PFO.")
    
    left_ventricle_ef: Optional[float] = Field(None, description="Left Ventricular Ejection Fraction in percentage.")
    
    aortic_valve: Optional[AorticValveSimplified] = Field(None, description="Details of the aortic valve.")

    overall_impression: Optional[str] = Field(None, description="Overall impression text from the report.")
