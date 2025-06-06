from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, validator, Field
import pytz
from sqlalchemy import Index, create_engine, Column, Integer, String, Boolean, ForeignKey, DateTime, Text, Float, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from passlib.context import CryptContext
from jose import JWTError, jwt
from typing import Optional, List, Union, Dict, Any
from enum import Enum
from datetime import datetime, time, timedelta, date, timezone
import uuid
import secrets
import string
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Form
from typing import Optional
from sqlalchemy.orm import joinedload

# JWT settings
SECRET_KEY = "gtZcWep!>oTJbF#TnQ%f>Oxn9pt'/{H;"
ALGORITHM = "HS256"
SECRET_ALLOCATOR_KEY = "Task_Allocater_BiD_Himanshu_Neha"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours
IST = pytz.timezone('Asia/Kolkata')
# Setup FastAPI
app = FastAPI(title="User Authentication and Task Management System")

# Setup database
SQLALCHEMY_DATABASE_URL = "sqlite:///./auth_app.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Client list
DEFAULT_CLIENTS = ["DND", "LOD", "FANTASIA", "BROADWAY", "MDB", "Rotomag", "XAU", "FINSYNC", "AMEY", "BiD"]
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://task.buildingindiadigital.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)
# Role enum
class UserRole(str, Enum):
    CLIENT = "client"
    ALLOCATOR = "allocator"
    EMPLOYEE = "employee"

# Task Status enum
class TaskStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"

# Database models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String)
    is_active = Column(Boolean, default=True)
    is_approved = Column(Boolean, default=False)  # New field for approval status
    reset_token = Column(String, nullable=True)
    reset_token_expires = Column(String, nullable=True)
    login_history = relationship("LoginHistory", back_populates="user")
    assigned_tasks = relationship("Task", back_populates="employee", foreign_keys="Task.employee_id")
    created_tasks = relationship("Task", back_populates="allocator", foreign_keys="Task.allocator_id")
    task_reports = relationship("TaskReport", back_populates="employee")
    payroll_records = relationship("PayrollRecord", back_populates="employee")
    salaries = relationship("EmployeeSalary", back_populates="employee")
    # Weekly sheet relationships
    created_sheets = relationship("WeeklySheet", foreign_keys="WeeklySheet.created_by", back_populates="creator")
    assigned_sheets = relationship("WeeklySheet", foreign_keys="WeeklySheet.assigned_to", back_populates="assignee")

class ApprovalRequest(Base):
    __tablename__ = "approval_requests"
    
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("users.id"))
    status = Column(String, default="pending")  # pending, approved, rejected
    requested_at = Column(DateTime, default=datetime.utcnow)
    responded_at = Column(DateTime, nullable=True)
    
    employee = relationship("User", foreign_keys=[employee_id])

class LoginHistory(Base):
    __tablename__ = "login_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    ip_address = Column(String)
    login_time = Column(DateTime, default=datetime.utcnow)
    user_agent = Column(String, nullable=True)
    
    user = relationship("User", back_populates="login_history")

class Client(Base):
    __tablename__ = "clients"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    tasks = relationship("Task", back_populates="client")

class Task(Base):
    __tablename__ = "tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, unique=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"))
    allocator_id = Column(Integer, ForeignKey("users.id"))
    employee_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    title = Column(String)
    description = Column(Text)
    status = Column(String, default=TaskStatus.PENDING, nullable=False)
    due_date = Column(DateTime)
    assigned_date = Column(DateTime, default=datetime.utcnow)
    completion_instructions = Column(Text, nullable=True)
    
    client = relationship("Client", back_populates="tasks")
    allocator = relationship("User", foreign_keys=[allocator_id], back_populates="created_tasks")
    employee = relationship("User", foreign_keys=[employee_id], back_populates="assigned_tasks")
    report = relationship("TaskReport", back_populates="task", uselist=False)

class TaskReport(Base):
    __tablename__ = "task_reports"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), unique=True)
    employee_id = Column(Integer, ForeignKey("users.id"))
    completion_description = Column(Text)
    hurdles_faced = Column(Text)
    completion_date = Column(DateTime, default=datetime.utcnow)
    allocator_feedback = Column(Text, nullable=True)
    hours_worked = Column(Float, default=0.0)
    work_location = Column(String, default="office")  # Add this new field
    
    task = relationship("Task", back_populates="report")
    employee = relationship("User", back_populates="task_reports")

class WorkSession(Base):
    __tablename__ = "work_sessions"
    __table_args__ = (
        Index('idx_work_session_employee', 'employee_id'),
        Index('idx_work_session_dates', 'clock_in', 'clock_out'),
    )
    
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("users.id"))
    clock_in = Column(DateTime, default=datetime.utcnow)
    clock_out = Column(DateTime, nullable=True)
    duration_minutes = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)
    ip_address = Column(String, nullable=True)
    device_info = Column(String, nullable=True)
    
    employee = relationship("User", backref="work_sessions")

class EmployeeSalary(Base):
    __tablename__ = "employee_salaries"
    
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("users.id"))
    monthly_salary = Column(Float)
    currency = Column(String, default="INR")
    effective_from = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    employee = relationship("User", back_populates="salaries")

class PayrollPeriod(Base):
    __tablename__ = "payroll_periods"
    __table_args__ = (
        Index('idx_payroll_period_dates', 'start_date', 'end_date'),
        Index('idx_payroll_period_status', 'status'),
        {'extend_existing': True}
    )
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    status = Column(String, default="draft")  # draft, processing, completed, locked
    created_by = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    locked_at = Column(DateTime, nullable=True)
    locked_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    creator = relationship("User", foreign_keys=[created_by])
    locker = relationship("User", foreign_keys=[locked_by])
    records = relationship("PayrollRecord", back_populates="period")

class PayrollRecord(Base):
    __tablename__ = "payroll_records"
    __table_args__ = (
        Index('idx_payroll_record_employee', 'employee_id'),
        Index('idx_payroll_record_period', 'period_id'),
        Index('idx_payroll_record_status', 'status'),
        {'extend_existing': True}
    )
    
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("users.id"))
    period_id = Column(Integer, ForeignKey("payroll_periods.id"))
    base_salary = Column(Float)
    total_minutes = Column(Float)
    expected_minutes = Column(Float)  # New field
    hourly_rate = Column(Float)
    overtime_minutes = Column(Float)
    overtime_rate = Column(Float)
    undertime_minutes = Column(Float, default=0.0)  # New field
    undertime_deduction = Column(Float, default=0.0)  # New field
    calculated_salary = Column(Float)
    currency = Column(String, default="INR")
    status = Column(String, default="pending", nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    employee = relationship("User", back_populates="payroll_records")
    period = relationship("PayrollPeriod", back_populates="records")

# NEW WEEKLY SHEET MODELS
class WeeklySheet(Base):
    __tablename__ = "weekly_sheets"
    
    id = Column(Integer, primary_key=True, index=True)
    sheet_id = Column(String, unique=True, index=True)
    month = Column(Integer)  # 1-12
    year = Column(Integer)
    created_by = Column(Integer, ForeignKey("users.id"))  # Allocator who created
    assigned_to = Column(Integer, ForeignKey("users.id"), nullable=True)  # Employee assigned (if any)
    is_template = Column(Boolean, default=False)  # True if it's allocator template, False if employee sheet
    status = Column(String, default="draft")  # draft, submitted, approved, rejected
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    submitted_at = Column(DateTime, nullable=True)
    
    creator = relationship("User", foreign_keys=[created_by], back_populates="created_sheets")
    assignee = relationship("User", foreign_keys=[assigned_to], back_populates="assigned_sheets")
    entries = relationship("WeeklySheetEntry", back_populates="sheet", cascade="all, delete-orphan")

class WeeklySheetEntry(Base):
    __tablename__ = "weekly_sheet_entries"
    
    id = Column(Integer, primary_key=True, index=True)
    sheet_id = Column(Integer, ForeignKey("weekly_sheets.id"))
    client_name = Column(String)  # DND, LOD, FANTASIA, etc.
    week_number = Column(Integer)  # 1-5
    posts_count = Column(Integer, default=0)
    reels_count = Column(Integer, default=0)
    story_description = Column(Text, nullable=True)
    is_topical_day = Column(Boolean, default=False)
    
    sheet = relationship("WeeklySheet", back_populates="entries")

# Create tables
Base.metadata.create_all(bind=engine)

# Initialize clients
def init_clients(db: Session):
    """Initialize default clients if they don't exist (only runs once)"""
    for client_name in DEFAULT_CLIENTS:
        existing_client = db.query(Client).filter(Client.name == client_name).first()
        if not existing_client:
            new_client = Client(name=client_name)
            db.add(new_client)
    db.commit()

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        # Initialize clients in database
        init_clients(db)
        yield db
    finally:
        db.close()

# Pydantic models
class UserBase(BaseModel):
    username: str
    email: EmailStr

class UserCreate(UserBase):
    password: str
    confirm_password: str
    role: UserRole
    secret_key: Optional[str] = None  # Add this field
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v
    
    @validator('secret_key')
    def validate_secret_key(cls, v, values):
        # Only validate secret key if role is ALLOCATOR
        if 'role' in values and values['role'] == UserRole.ALLOCATOR:
            if not v:
                raise ValueError('Secret key is required for allocator registration')
            if v != SECRET_ALLOCATOR_KEY:
                raise ValueError('Invalid secret key')
        return v

class UserLogin(BaseModel):
    username: str
    password: str
    role: Optional[UserRole] = None

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class PasswordResetRequest(BaseModel):
    username: str

class ResetTokenResponse(BaseModel):
    reset_token: str
    message: str

class PasswordReset(BaseModel):
    token: str
    new_password: str
    confirm_password: str

class ApprovalAction(BaseModel):
    employee_id: int
    action: str  # "approve" or "reject"
    
    @validator('action')
    def validate_action(cls, v):
        if v not in ["approve", "reject"]:
            raise ValueError('Action must be either "approve" or "reject"')
        return v

class UserResponse(UserBase):
    id: int
    role: str
    is_active: bool
    is_approved: bool
    
    class Config:
        orm_mode = True

class LoginHistoryResponse(BaseModel):
    id: int
    ip_address: str
    login_time: datetime
    user_agent: Optional[str] = None
    
    class Config:
        orm_mode = True

class UserDetailResponse(UserResponse):
    login_history: List[LoginHistoryResponse] = []
    
    class Config:
        orm_mode = True

class ClientResponse(BaseModel):
    id: int
    name: str
    
    class Config:
        orm_mode = True

class TaskCreate(BaseModel):
    client_id: int
    employee_id: int
    title: str
    description: str
    due_date: datetime
    completion_instructions: Optional[str] = None

class TaskUpdate(BaseModel):
    status: TaskStatus
    allocator_feedback: Optional[str] = None

class TaskReportCreate(BaseModel):
    completion_description: str
    hurdles_faced: str
    hours_worked: float
    work_location: str = "office"  # Add this new field with a default value

class TaskReportResponse(BaseModel):
    id: int
    completion_description: str
    hurdles_faced: str
    completion_date: datetime
    allocator_feedback: Optional[str] = None
    hours_worked: float
    work_location: str
    
    class Config:
        orm_mode = True
class ClientCreate(BaseModel):
    name: str
    
    @validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Client name cannot be empty')
        # Convert to uppercase and remove extra spaces
        return v.strip().upper()
class WorkSessionCreate(BaseModel):
    notes: Optional[str] = None

class WorkSessionUpdate(BaseModel):
    notes: Optional[str] = None

class WorkSessionResponse(BaseModel):
    id: int
    employee_id: int
    clock_in: datetime
    clock_out: Optional[datetime] = None
    duration_minutes: Optional[float] = None
    notes: Optional[str] = None
    
    class Config:
        orm_mode = True

class EmployeeSalaryBase(BaseModel):
    employee_id: int
    monthly_salary: float
    currency: Optional[str] = "INR"

class EmployeeSalaryCreate(EmployeeSalaryBase):
    pass

class EmployeeSalaryResponse(BaseModel):
    id: int
    employee_id: int
    monthly_salary: float
    currency: str
    effective_from: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True
        from_attributes = True  # This is the key fix for Pydantic v2

class EmployeeWithSalaryResponse(UserResponse):
    current_salary: Optional[EmployeeSalaryResponse] = None
    salary_history: Optional[List[EmployeeSalaryResponse]] = Field(None, exclude=True)
    
    @validator('salary_history', pre=True, always=True)
    def set_salary_history(cls, v, values, **kwargs):
        if 'id' in values:
            # This assumes the salary history will be added during object creation
            return v or []
        return []

class TaskResponse(BaseModel):
    id: int
    task_id: str
    client: ClientResponse
    allocator: UserResponse
    employee: Optional[UserResponse] = None
    title: str
    description: str
    status: str
    due_date: datetime
    assigned_date: datetime
    completion_instructions: Optional[str] = None
    report: Optional[TaskReportResponse] = None
    
    class Config:
        orm_mode = True

class PayrollPeriodCreate(BaseModel):
    name: str
    start_date: datetime
    end_date: datetime

class PayrollPeriodResponse(BaseModel):
    id: int
    name: str
    start_date: datetime
    end_date: datetime
    status: str
    created_at: datetime
    
    class Config:
        orm_mode = True

class PayrollRecordResponse(BaseModel):
    id: int
    employee_id: int
    period_id: int
    period: Optional[PayrollPeriodResponse] = None
    base_salary: float
    total_minutes: float
    expected_minutes: float  # New field
    hourly_rate: float
    overtime_minutes: float
    overtime_rate: float
    undertime_minutes: float  # New field
    undertime_deduction: float  # New field
    calculated_salary: float
    currency: str
    status: str
    created_at: datetime
    
    class Config:
        orm_mode = True

class PayrollCalculationParams(BaseModel):
    standard_monthly_hours: float = Field(160, description="Standard monthly hours (default 160)")
    overtime_multiplier: float = Field(1.5, description="Overtime pay multiplier (default 1.5)")
    apply_deductions: bool = Field(True, description="Apply deductions for undertime (default True)")

# NEW WEEKLY SHEET PYDANTIC MODELS
class WeeklySheetEntryCreate(BaseModel):
    client_name: str
    week_number: int
    posts_count: int = 0
    reels_count: int = 0
    story_description: Optional[str] = None
    is_topical_day: bool = False

class WeeklySheetEntryUpdate(BaseModel):
    posts_count: Optional[int] = None
    reels_count: Optional[int] = None
    story_description: Optional[str] = None

class WeeklySheetEntryResponse(BaseModel):
    id: int
    client_name: str
    week_number: int
    posts_count: int
    reels_count: int
    story_description: Optional[str]
    is_topical_day: bool
    
    class Config:
        orm_mode = True

class WeeklySheetCreate(BaseModel):
    month: int
    year: int
    assigned_to: Optional[int] = None  # Employee ID if assigning to specific employee
    entries: List[WeeklySheetEntryCreate]

class WeeklySheetUpdate(BaseModel):
    entries: List[WeeklySheetEntryUpdate]

class WeeklySheetResponse(BaseModel):
    id: int
    sheet_id: str
    month: int
    year: int
    created_by: int
    assigned_to: Optional[int]
    is_template: bool
    status: str
    created_at: datetime
    updated_at: datetime
    submitted_at: Optional[datetime]
    creator: UserResponse
    assignee: Optional[UserResponse]
    entries: List[WeeklySheetEntryResponse]
    
    class Config:
        orm_mode = True

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def generate_reset_token():
    """Generate a secure, unique reset token"""
    # Create a random string with letters, digits, and special characters
    alphabet = string.ascii_letters + string.digits + "@#$%^&*"
    token = ''.join(secrets.choice(alphabet) for _ in range(12))
    return token

def generate_task_id():
    """Generate a unique task ID with prefix Task- and a hexadecimal code"""
    hex_code = uuid.uuid4().hex[:8]
    return f"Task-{hex_code}"

# NEW HELPER FUNCTIONS FOR WEEKLY SHEETS
def generate_sheet_id():
    """Generate a unique sheet ID"""
    return f"SHEET-{uuid.uuid4().hex[:8]}"

def get_current_month_year():
    """Get current month and year"""
    now = datetime.now(IST)
    return now.month, now.year

def get_user(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def get_user_by_id(db: Session, user_id: int):
    return db.query(User).filter(User.id == user_id).first()

def get_client_by_id(db: Session, client_id: int):
    return db.query(Client).filter(Client.id == client_id).first()

def get_employees(db: Session):
    return db.query(User).filter(User.role == UserRole.EMPLOYEE).all()

def get_task_by_id(db: Session, task_id: int):
    return db.query(Task).filter(Task.id == task_id).first()

def authenticate_user(db: Session, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = get_user(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    return user

def check_allocator_role(current_user: User = Depends(get_current_user)):
    if current_user.role != UserRole.ALLOCATOR:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to perform this action. Allocator role required."
        )
    return current_user

def check_employee_role(current_user: User = Depends(get_current_user)):
    if current_user.role != UserRole.EMPLOYEE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to perform this action. Employee role required."
        )
    return current_user

def ensure_timezone_aware(dt, timezone=IST):
    """Ensure a datetime object has timezone information."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone)
    return dt

def now_with_timezone():
    """Get current datetime with IST timezone."""
    return datetime.now(IST)

def get_employee_salary(db: Session, employee_id: int):
    return db.query(EmployeeSalary).filter(
        EmployeeSalary.employee_id == employee_id
    ).order_by(EmployeeSalary.effective_from.desc()).first()

def create_or_update_salary(db: Session, salary_data: EmployeeSalaryCreate):
    # Create new salary record (maintains history)
    new_salary = EmployeeSalary(
        employee_id=salary_data.employee_id,
        monthly_salary=salary_data.monthly_salary,
        currency=salary_data.currency
    )
    db.add(new_salary)
    db.commit()
    db.refresh(new_salary)
    return new_salary

def calculate_employee_payroll(
    db: Session,
    employee_id: int,
    period_id: int,
    standard_hours: float = 160,
    overtime_multiplier: float = 1.5,
    apply_deductions: bool = True
) -> PayrollRecord:
    """
    Calculate payroll for an employee in a specific period with deductions for undertime
    Returns PayrollRecord object ready to be committed
    """
    # First get the period to access its dates
    period = db.query(PayrollPeriod).filter(PayrollPeriod.id == period_id).first()
    if not period:
        raise ValueError(f"Payroll period {period_id} not found")

    # Now we can use period.start_date and period.end_date
    start_date = period.start_date
    end_date = period.end_date

    # Get employee and current salary
    employee = db.query(User).filter(
        User.id == employee_id,
        User.role == UserRole.EMPLOYEE
    ).first()
    if not employee:
        raise ValueError(f"Employee {employee_id} not found")

    salary = get_employee_salary(db, employee_id)
    if not salary:
        raise ValueError(f"No salary record for employee {employee_id}")

    # FIXED: Get all completed work sessions that overlap with the period
    # A session overlaps with the period if:
    # 1. It started before or during the period AND
    # 2. It ended during or after the period
    sessions = db.query(WorkSession).filter(
        WorkSession.employee_id == employee_id,
        WorkSession.clock_in <= end_date,        # Started before period ended
        WorkSession.clock_out >= start_date,     # Ended after period started
        WorkSession.clock_out.isnot(None)        # Session is completed
    ).all()

    # For sessions that partially overlap with the period, only count the overlapping time
    total_minutes = 0
    for session in sessions:
        # Get the actual start and end times within the period
        effective_start = max(session.clock_in, start_date)
        effective_end = min(session.clock_out, end_date)
        
        # Calculate the duration within the period
        duration_minutes = (effective_end - effective_start).total_seconds() / 60
        total_minutes += max(0, duration_minutes)  # Ensure no negative durations
    
    # Calculate standard work minutes for the period
    standard_minutes = standard_hours * 60
    
    # Calculate overtime and undertime
    overtime_minutes = max(0, total_minutes - standard_minutes)
    undertime_minutes = max(0, standard_minutes - total_minutes) if apply_deductions else 0
    
    # Calculate rates
    hourly_rate = salary.monthly_salary / standard_hours
    overtime_rate = hourly_rate * overtime_multiplier
    
    # Calculate base, overtime, and deductions
    base_salary = salary.monthly_salary
    overtime_pay = (overtime_minutes / 60) * overtime_rate
    undertime_deduction = (undertime_minutes / 60) * hourly_rate if apply_deductions else 0
    
    # Calculate final salary with deductions for undertime
    calculated_salary = base_salary + overtime_pay - undertime_deduction

    # Create payroll record with new fields
    record = PayrollRecord(
        employee_id=employee_id,
        period_id=period_id,
        base_salary=base_salary,
        total_minutes=total_minutes,
        expected_minutes=standard_minutes,
        hourly_rate=hourly_rate,
        overtime_minutes=overtime_minutes,
        overtime_rate=overtime_rate,
        undertime_minutes=undertime_minutes,
        undertime_deduction=undertime_deduction,
        calculated_salary=calculated_salary,
        currency=salary.currency
    )
    
    return record

def generate_payroll_for_period(
    db: Session,
    period_id: int,
    params: PayrollCalculationParams
) -> List[PayrollRecord]:
    """Generate payroll records for all employees in a period"""
    period = db.query(PayrollPeriod).get(period_id)
    if not period:
        raise ValueError("Payroll period not found")
    
    if period.status == "locked":
        raise ValueError("Payroll period is locked")

    # Mark as processing
    period.status = "processing"
    db.commit()

    try:
        employees = db.query(User).filter(User.role == UserRole.EMPLOYEE).all()
        records = []
        
        for employee in employees:
            try:
                record = calculate_employee_payroll(
                    db,
                    employee.id,
                    period_id,
                    params.standard_monthly_hours,
                    params.overtime_multiplier,
                    params.apply_deductions  # Pass the new parameter
                )
                db.add(record)
                records.append(record)
            except Exception as e:
                db.rollback()
                # Log error but continue with other employees
                print(f"Error processing employee {employee.id}: {str(e)}")
                continue
        
        db.commit()
        period.status = "completed"
        db.commit()
        
        return records
    
    except Exception as e:
        db.rollback()
        period.status = "draft"
        db.commit()
        raise

# API Endpoints
@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    # Check if username already exists (including rejected employees)
    db_user = get_user(db, username=user.username)
    if db_user:
        # Check if this user was rejected
        if db_user.role == UserRole.EMPLOYEE and not db_user.is_approved:
            # Check if there's a rejected approval request
            rejected_request = db.query(ApprovalRequest).filter(
                ApprovalRequest.employee_id == db_user.id,
                ApprovalRequest.status == "rejected"
            ).first()
            
            if rejected_request:
                raise HTTPException(
                    status_code=400, 
                    detail="This username was previously rejected and cannot be reused"
                )
        
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Check if email already exists (including rejected employees)
    db_user_email = get_user_by_email(db, email=user.email)
    if db_user_email:
        # Check if this user was rejected
        if db_user_email.role == UserRole.EMPLOYEE and not db_user_email.is_approved:
            # Check if there's a rejected approval request
            rejected_request = db.query(ApprovalRequest).filter(
                ApprovalRequest.employee_id == db_user_email.id,
                ApprovalRequest.status == "rejected"
            ).first()
            
            if rejected_request:
                raise HTTPException(
                    status_code=400, 
                    detail="This email was previously rejected and cannot be reused"
                )
        
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Validate secret key for allocator registration
    if user.role == UserRole.ALLOCATOR:
        if not user.secret_key or user.secret_key != SECRET_ALLOCATOR_KEY:
            raise HTTPException(
                status_code=400, 
                detail="Invalid or missing secret key for allocator registration"
            )
    
    # Create new user
    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        role=user.role,
        is_approved=user.role != UserRole.EMPLOYEE  # Only employees need approval
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Create approval request if the user is an employee
    if user.role == UserRole.EMPLOYEE:
        approval_request = ApprovalRequest(employee_id=db_user.id)
        db.add(approval_request)
        db.commit()
    
    return db_user
@app.post("/login", response_model=Token)
def login_for_access_token(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(), 
    role: Optional[UserRole] = Form(None),
    db: Session = Depends(get_db)
):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify role if provided
    if role and user.role != role:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You don't have {role} privileges",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check approval status for employees
    if user.role == UserRole.EMPLOYEE and not user.is_approved:
        # Create a new approval request if the last one was rejected or doesn't exist
        existing_request = db.query(ApprovalRequest).filter(
            ApprovalRequest.employee_id == user.id,
            ApprovalRequest.status == "pending"
        ).first()
        
        if not existing_request:
            approval_request = ApprovalRequest(employee_id=user.id)
            db.add(approval_request)
            db.commit()
            
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account is pending approval by an allocator.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Record login history
    client_ip = request.client.host
    user_agent = request.headers.get("user-agent", "Unknown")
    
    login_record = LoginHistory(
        user_id=user.id,
        ip_address=client_ip,
        user_agent=user_agent
    )
    
    db.add(login_record)
    db.commit()
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

# Get pending approval requests for allocators
@app.get("/allocators/pending-approvals", response_model=List[Dict])
def get_pending_approvals(
    db: Session = Depends(get_db),
    current_user: User = Depends(check_allocator_role)
):
    # Get all pending approval requests
    pending_requests = db.query(ApprovalRequest).filter(
        ApprovalRequest.status == "pending"
    ).order_by(ApprovalRequest.requested_at.desc()).all()
    
    results = []
    for request in pending_requests:
        employee = db.query(User).filter(User.id == request.employee_id).first()
        if employee:
            results.append({
                "request_id": request.id,
                "employee_id": employee.id,
                "employee_username": employee.username,
                "employee_email": employee.email,
                "requested_at": request.requested_at
            })
    
    return results

# Endpoint for allocators to approve or reject employee registrations
@app.post("/allocators/employee-approval")
def process_employee_approval(
    approval: ApprovalAction,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_allocator_role)
):
    # Find the employee
    employee = db.query(User).filter(User.id == approval.employee_id).first()
    if not employee or employee.role != UserRole.EMPLOYEE:
        raise HTTPException(status_code=404, detail="Employee not found")
    
    # Find pending approval request
    approval_request = db.query(ApprovalRequest).filter(
        ApprovalRequest.employee_id == employee.id,
        ApprovalRequest.status == "pending"
    ).first()
    
    if not approval_request:
        raise HTTPException(status_code=404, detail="No pending approval request found")
    
    # Process the approval action
    if approval.action == "approve":
        employee.is_approved = True
        approval_request.status = "approved"
    elif approval.action == "reject":
        approval_request.status = "rejected"
    else:
        raise HTTPException(status_code=400, detail="Invalid action. Use 'approve' or 'reject'")
    
    approval_request.responded_at = datetime.utcnow()
    db.commit()
    
    return {"message": f"Employee {approval.action}d successfully"}

@app.post("/password-reset-request", response_model=ResetTokenResponse)
def request_password_reset(request: PasswordResetRequest, db: Session = Depends(get_db)):
    # Find user by username
    user = get_user(db, username=request.username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Generate unique reset token
    reset_token = generate_reset_token()
    token_expires = datetime.utcnow() + timedelta(hours=1)
    
    # Update user with reset token
    user.reset_token = reset_token
    user.reset_token_expires = token_expires.isoformat()
    db.commit()
    
    # Return the token directly (no email sending)
    return {
        "reset_token": reset_token,
        "message": "Use this token to reset your password. This token will expire in 1 hour."
    }

@app.post("/reset-password")
def reset_password(reset_data: PasswordReset, db: Session = Depends(get_db)):
    # Find user with the token
    user = db.query(User).filter(User.reset_token == reset_data.token).first()
    
    if not user:
        raise HTTPException(status_code=400, detail="Invalid token")
    
    # Check if token is expired
    token_expires = datetime.fromisoformat(user.reset_token_expires)
    if datetime.utcnow() > token_expires:
        raise HTTPException(status_code=400, detail="Token has expired")
    
    # Update password
    user.hashed_password = get_password_hash(reset_data.new_password)
    user.reset_token = None
    user.reset_token_expires = None
    db.commit()
    
    return {"message": "Password has been reset successfully"}

@app.get("/users/me", response_model=UserResponse)
def get_current_user_info(current_user: User = Depends(get_current_user)):
    return current_user

@app.get("/users/me/detail", response_model=UserDetailResponse)
def get_current_user_detail(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Get user with login history
    return current_user

@app.get("/users/me/login-history", response_model=List[LoginHistoryResponse])
def get_login_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Get login history for current user
    return current_user.login_history

@app.get("/")
def root():
    return {"message": "Authentication and Task Management API is running. Visit /docs for OpenAPI documentation."}

# Client Management
@app.get("/clients", response_model=List[ClientResponse])
def get_all_clients(db: Session = Depends(get_db), current_user: User = Depends(check_allocator_role)):
    # Only allocators can see the client list
    clients = db.query(Client).all()
    return clients

# Employee Management for Allocators
@app.get("/employees", response_model=List[EmployeeWithSalaryResponse])
def get_all_employees(
    include_rejected: bool = False,  # Query parameter to optionally include rejected employees
    db: Session = Depends(get_db), 
    current_user: User = Depends(check_allocator_role)
):
    """
    Get all employees with their current salary information
    
    Args:
        include_rejected: If True, includes employees who have been rejected. Default: False
    """
    
    # Start with base query for all employees
    query = db.query(User).filter(User.role == UserRole.EMPLOYEE)
    
    # If not including rejected employees, exclude them
    if not include_rejected:
        # Get employee IDs that have been rejected
        rejected_employee_ids = db.query(ApprovalRequest.employee_id).filter(
            ApprovalRequest.status == "rejected"
        ).subquery()
        
        # Exclude rejected employees from the query
        query = query.filter(~User.id.in_(rejected_employee_ids))
    
    # Execute the query
    employees = query.all()
    
    # Enhance response with salary data
    result = []
    for employee in employees:
        employee_data = employee.__dict__.copy()  # Use copy() to avoid modifying original
        salary = get_employee_salary(db, employee.id)
        employee_data["current_salary"] = salary
        result.append(employee_data)
    
    return result

@app.post("/employees/{employee_id}/salary", response_model=EmployeeSalaryResponse)
def set_employee_salary(
    employee_id: int,
    salary_data: EmployeeSalaryCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_allocator_role)
):
    """Set or update an employee's salary"""
    # Verify employee exists and is actually an employee
    employee = db.query(User).filter(
        User.id == employee_id,
        User.role == UserRole.EMPLOYEE
    ).first()
    
    if not employee:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Employee not found"
        )
    
    # Verify the employee_id in payload matches URL
    if employee_id != salary_data.employee_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Employee ID mismatch"
        )
    
    # Create new salary record
    try:
        return create_or_update_salary(db, salary_data)
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error setting salary: {str(e)}"
        )

@app.get("/employees/{employee_id}/salary-history", response_model=List[EmployeeSalaryResponse])
def get_employee_salary_history(
    employee_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_allocator_role)
):
    """Get complete salary history for an employee"""
    employee = db.query(User).filter(
        User.id == employee_id,
        User.role == UserRole.EMPLOYEE
    ).first()
    
    if not employee:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Employee not found"
        )
    
    return db.query(EmployeeSalary).filter(
        EmployeeSalary.employee_id == employee_id
    ).order_by(EmployeeSalary.effective_from.desc()).all()

# Task Management for Allocators
@app.post("/tasks", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
def create_task(
    task_data: TaskCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_allocator_role)
):
    # Check if client exists
    client = get_client_by_id(db, task_data.client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Check if employee exists
    employee = get_user_by_id(db, task_data.employee_id)
    if not employee or employee.role != UserRole.EMPLOYEE:
        raise HTTPException(status_code=404, detail="Employee not found")
    
    # Create task - still record which allocator created it
    new_task = Task(
        task_id=generate_task_id(),
        client_id=task_data.client_id,
        allocator_id=current_user.id,  # Still track who created it
        employee_id=task_data.employee_id,
        title=task_data.title,
        description=task_data.description,
        status=TaskStatus.PENDING,
        due_date=task_data.due_date,
        completion_instructions=task_data.completion_instructions
    )
    
    db.add(new_task)
    db.commit()
    db.refresh(new_task)
    
    return new_task

@app.get("/tasks", response_model=List[TaskResponse])
def get_all_tasks(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    status: Optional[TaskStatus] = None
):
    query = db.query(Task)
    
    # For allocators, show all tasks (not just their own)
    if current_user.role == UserRole.EMPLOYEE:
        query = query.filter(Task.employee_id == current_user.id)
    
    # Filter by status if provided
    if status:
        query = query.filter(Task.status == status)
    
    return query.all()

@app.get("/tasks/{task_id}", response_model=TaskResponse)
def get_task(
    task_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    task = get_task_by_id(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Check permission - allow any allocator to see any task
    if current_user.role == UserRole.EMPLOYEE and task.employee_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this task")
    
    return task

@app.put("/tasks/{task_id}/status", response_model=TaskResponse)
def update_task_status(
    task_id: int,
    task_update: TaskUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_allocator_role)
):
    task = get_task_by_id(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # No need to check if allocator owns this task - allow any allocator to update any task
    
    # Update task status
    task.status = task_update.status
    
    # If there's a report and feedback is provided, update it
    if task.report and task_update.allocator_feedback:
        task.report.allocator_feedback = task_update.allocator_feedback
    
    db.commit()
    db.refresh(task)
    
    return task

@app.get("/allocators/employee/{employee_id}/timesheets", response_model=List[Dict])
def get_employee_timesheets_for_allocator(
    employee_id: int,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_allocator_role)
):
    """Get timesheet data (work sessions) for a specific employee (allocator view)"""
    # Check if employee exists
    employee = db.query(User).filter(
        User.id == employee_id,
        User.role == UserRole.EMPLOYEE
    ).first()
    
    if not employee:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Employee not found"
        )
    
    # Query work sessions
    query = db.query(WorkSession).filter(
        WorkSession.employee_id == employee_id,
        WorkSession.clock_out.isnot(None)  # Only completed sessions
    )
    
    # Filter by date range if provided
    if start_date:
        start_datetime = datetime.combine(start_date, datetime.min.time())
        query = query.filter(WorkSession.clock_in >= start_datetime)
    
    if end_date:
        end_datetime = datetime.combine(end_date, datetime.max.time())
        query = query.filter(WorkSession.clock_in <= end_datetime)
    
    # Order by clock_in time (newest first)
    sessions = query.order_by(WorkSession.clock_in.desc()).all()
    
    # Group by date
    timesheets = {}
    for session in sessions:
        # Get the date in string format
        date_str = session.clock_in.date().isoformat()
        
        if date_str not in timesheets:
            timesheets[date_str] = {
                "date": date_str,
                "total_minutes": 0,
                "sessions": []
            }
        
        # Add session details
        session_data = {
            "id": session.id,
            "clock_in": session.clock_in,
            "clock_out": session.clock_out,
            "duration_minutes": session.duration_minutes or 0,
            "notes": session.notes
        }
        
        timesheets[date_str]["sessions"].append(session_data)
        timesheets[date_str]["total_minutes"] += session.duration_minutes or 0
    
    # Convert to list sorted by date (newest first)
    result = list(timesheets.values())
    result.sort(key=lambda x: x["date"], reverse=True)
    
    return result

@app.delete("/allocators/employee/{employee_id}/timesheets")
def clear_employee_timesheets(
    employee_id: int,
    before_date: date,  # Delete sessions before this date
    db: Session = Depends(get_db),
    current_user: User = Depends(check_allocator_role)
):
    """Delete timesheet data (work sessions) for a specific employee before a specified date"""
    # Check if employee exists
    employee = db.query(User).filter(
        User.id == employee_id,
        User.role == UserRole.EMPLOYEE
    ).first()
    
    if not employee:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Employee not found"
        )
    
    # Calculate the cutoff datetime (beginning of the specified date)
    cutoff_datetime = datetime.combine(before_date, datetime.min.time())
    
    # Find sessions to delete
    sessions_to_delete = db.query(WorkSession).filter(
        WorkSession.employee_id == employee_id,
        WorkSession.clock_in < cutoff_datetime
    )
    
    # Count sessions that will be deleted
    count = sessions_to_delete.count()
    
    if count == 0:
        return {"message": "No timesheet entries found before the specified date"}
    
    # Delete the sessions
    sessions_to_delete.delete(synchronize_session=False)
    db.commit()
    
    return {
        "message": f"Successfully deleted {count} timesheet entries before {before_date.isoformat()}",
        "deleted_count": count
    }

@app.delete("/employees/timesheets")
def clear_employee_own_timesheets(
    before_date: date,  # Delete sessions before this date
    db: Session = Depends(get_db),
    current_user: User = Depends(check_employee_role)
):
    """Delete the current employee's own timesheet data (work sessions) before a specified date"""
    # Calculate the cutoff datetime (beginning of the specified date)
    cutoff_datetime = datetime.combine(before_date, datetime.min.time())
    
    # Find sessions to delete - only the employee's own sessions
    sessions_to_delete = db.query(WorkSession).filter(
        WorkSession.employee_id == current_user.id,
        WorkSession.clock_in < cutoff_datetime
    )
    
    # Count sessions that will be deleted
    count = sessions_to_delete.count()
    
    if count == 0:
        return {"message": "No timesheet entries found before the specified date"}
    
    # Delete the sessions
    sessions_to_delete.delete(synchronize_session=False)
    db.commit()
    
    return {
        "message": f"Successfully deleted {count} timesheet entries before {before_date.isoformat()}",
        "deleted_count": count
    }

# Task reporting for employees
@app.post("/tasks/{task_id}/report", response_model=TaskReportResponse)
def submit_task_report(
    task_id: int,
    report_data: TaskReportCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_employee_role)
):
    task = get_task_by_id(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Check if employee is assigned to this task
    if task.employee_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to report on this task")
    
    # Check if report already exists
    if task.report:
        raise HTTPException(status_code=400, detail="Report already submitted for this task")
    
    # Create task report with the new work_location field
    new_report = TaskReport(
        task_id=task.id,
        employee_id=current_user.id,
        completion_description=report_data.completion_description,
        hurdles_faced=report_data.hurdles_faced,
        hours_worked=report_data.hours_worked,
        work_location=report_data.work_location  # Add this line
    )
    
    # Update task status to completed
    task.status = TaskStatus.COMPLETED
    
    db.add(new_report)
    db.commit()
    db.refresh(new_report)
    
    return new_report

# Endpoint for allocators to get tasks by client
@app.get("/tasks/client/{client_id}", response_model=List[TaskResponse])
def get_tasks_by_client(
    client_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_allocator_role)
):
    # Check if client exists
    client = get_client_by_id(db, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Get all tasks for this client (not just current allocator's tasks)
    tasks = db.query(Task).filter(
        Task.client_id == client_id
    ).all()
    
    return tasks

# NEW WEEKLY SHEET ENDPOINTS

@app.post("/allocators/weekly-sheets", response_model=WeeklySheetResponse)
def create_weekly_sheet(
    sheet_data: WeeklySheetCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_allocator_role)
):
    """Create a new weekly sheet (allocator creates template or assigns to employee)"""
    
    # Validate month and year
    if not (1 <= sheet_data.month <= 12):
        raise HTTPException(status_code=400, detail="Month must be between 1 and 12")
    
    # Check if employee exists (if assigned)
    if sheet_data.assigned_to:
        employee = db.query(User).filter(
            User.id == sheet_data.assigned_to,
            User.role == UserRole.EMPLOYEE
        ).first()
        if not employee:
            raise HTTPException(status_code=404, detail="Employee not found")
    
    # Check if sheet already exists for this month/year/employee combination
    existing_sheet = db.query(WeeklySheet).filter(
        WeeklySheet.month == sheet_data.month,
        WeeklySheet.year == sheet_data.year,
        WeeklySheet.assigned_to == sheet_data.assigned_to,
        WeeklySheet.created_by == current_user.id
    ).first()
    
    if existing_sheet:
        raise HTTPException(
            status_code=400, 
            detail="Sheet already exists for this month/year/employee combination"
        )
    
    # Create the sheet
    new_sheet = WeeklySheet(
        sheet_id=generate_sheet_id(),
        month=sheet_data.month,
        year=sheet_data.year,
        created_by=current_user.id,
        assigned_to=sheet_data.assigned_to,
        is_template=sheet_data.assigned_to is None
    )
    
    db.add(new_sheet)
    db.commit()
    db.refresh(new_sheet)
    
    # Create entries
    for entry_data in sheet_data.entries:
        # Validate client name
        if entry_data.client_name not in DEFAULT_CLIENTS:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid client name. Must be one of: {', '.join(DEFAULT_CLIENTS)}"
            )
        
        # Validate week number
        if not (1 <= entry_data.week_number <= 5):
            raise HTTPException(status_code=400, detail="Week number must be between 1 and 5")
        
        entry = WeeklySheetEntry(
            sheet_id=new_sheet.id,
            client_name=entry_data.client_name,
            week_number=entry_data.week_number,
            posts_count=entry_data.posts_count,
            reels_count=entry_data.reels_count,
            story_description=entry_data.story_description,
            is_topical_day=entry_data.is_topical_day
        )
        db.add(entry)
    
    db.commit()
    db.refresh(new_sheet)
    
    return new_sheet

@app.get("/allocators/weekly-sheets", response_model=List[WeeklySheetResponse])
def get_allocator_weekly_sheets(
    month: Optional[int] = None,
    year: Optional[int] = None,
    employee_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_allocator_role)
):
    """Get weekly sheets created by the allocator"""
    
    query = db.query(WeeklySheet).filter(WeeklySheet.created_by == current_user.id)
    
    # Filter by month if provided
    if month:
        query = query.filter(WeeklySheet.month == month)
    
    # Filter by year if provided
    if year:
        query = query.filter(WeeklySheet.year == year)
    
    # Filter by employee if provided
    if employee_id:
        query = query.filter(WeeklySheet.assigned_to == employee_id)
    
    return query.order_by(WeeklySheet.created_at.desc()).all()

@app.get("/allocators/weekly-sheets/{sheet_id}", response_model=WeeklySheetResponse)
def get_weekly_sheet_detail(
    sheet_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_allocator_role)
):
    """Get detailed view of a specific weekly sheet"""
    
    sheet = db.query(WeeklySheet).filter(WeeklySheet.sheet_id == sheet_id).first()
    if not sheet:
        raise HTTPException(status_code=404, detail="Sheet not found")
    
    # Check permission - allow any allocator to see any sheet
    if sheet.created_by != current_user.id and sheet.status != "submitted":
        raise HTTPException(status_code=403, detail="Not authorized to view this sheet")
    
    return sheet

@app.put("/allocators/weekly-sheets/{sheet_id}", response_model=WeeklySheetResponse)
def update_weekly_sheet(
    sheet_id: str,
    updates: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(check_allocator_role)
):
    """Update a weekly sheet (allocator can edit any sheet)"""
    
    sheet = db.query(WeeklySheet).filter(WeeklySheet.sheet_id == sheet_id).first()
    if not sheet:
        raise HTTPException(status_code=404, detail="Sheet not found")
    
    # Update entries if provided
    if "entries" in updates:
        for entry_update in updates["entries"]:
            if "id" in entry_update:
                entry = db.query(WeeklySheetEntry).filter(
                    WeeklySheetEntry.id == entry_update["id"],
                    WeeklySheetEntry.sheet_id == sheet.id
                ).first()
                
                if entry:
                    if "posts_count" in entry_update:
                        entry.posts_count = entry_update["posts_count"]
                    if "reels_count" in entry_update:
                        entry.reels_count = entry_update["reels_count"]
                    if "story_description" in entry_update:
                        entry.story_description = entry_update["story_description"]
    
    # Update sheet status if provided
    if "status" in updates:
        sheet.status = updates["status"]
    
    sheet.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(sheet)
    
    return sheet

@app.get("/allocators/employee-sheets/submitted", response_model=List[WeeklySheetResponse])
def get_submitted_employee_sheets(
    month: Optional[int] = None,
    year: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_allocator_role)
):
    """Get all submitted employee sheets for review"""
    
    query = db.query(WeeklySheet).filter(
        WeeklySheet.status == "submitted",
        WeeklySheet.is_template == False
    )
    
    # Filter by month if provided
    if month:
        query = query.filter(WeeklySheet.month == month)
    
    # Filter by year if provided
    if year:
        query = query.filter(WeeklySheet.year == year)
    
    return query.order_by(WeeklySheet.submitted_at.desc()).all()

@app.post("/allocators/auto-generate-monthly-sheets")
def auto_generate_monthly_sheets(
    month: int,
    year: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_allocator_role)
):
    """Auto-generate weekly sheets for all employees for a specific month"""
    
    if not (1 <= month <= 12):
        raise HTTPException(status_code=400, detail="Month must be between 1 and 12")
    
    # Get all active employees
    employees = db.query(User).filter(
        User.role == UserRole.EMPLOYEE,
        User.is_active == True,
        User.is_approved == True
    ).all()
    
    created_sheets = []
    
    for employee in employees:
        # Check if sheet already exists
        existing_sheet = db.query(WeeklySheet).filter(
            WeeklySheet.month == month,
            WeeklySheet.year == year,
            WeeklySheet.assigned_to == employee.id
        ).first()
        
        if existing_sheet:
            continue  # Skip if already exists
        
        # Create sheet for employee
        new_sheet = WeeklySheet(
            sheet_id=generate_sheet_id(),
            month=month,
            year=year,
            created_by=current_user.id,
            assigned_to=employee.id,
            is_template=False
        )
        
        db.add(new_sheet)
        db.commit()
        db.refresh(new_sheet)
        
        # Create default entries for all clients and weeks
        for client in DEFAULT_CLIENTS:
            for week in range(1, 6):  # Weeks 1-5
                entry = WeeklySheetEntry(
                    sheet_id=new_sheet.id,
                    client_name=client,
                    week_number=week,
                    posts_count=0,
                    reels_count=0,
                    story_description="COLLAGE + WTSAP STORY",
                    is_topical_day=False
                )
                db.add(entry)
        
        db.commit()
        created_sheets.append({
            "employee_id": employee.id,
            "employee_name": employee.username,
            "sheet_id": new_sheet.sheet_id
        })
    
    return {
        "message": f"Created {len(created_sheets)} sheets for {month}/{year}",
        "created_sheets": created_sheets
    }

# EMPLOYEE WEEKLY SHEET ENDPOINTS

@app.get("/employees/weekly-sheets", response_model=List[WeeklySheetResponse])
def get_employee_weekly_sheets(
    month: Optional[int] = None,
    year: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_employee_role)
):
    """Get weekly sheets assigned to the employee"""
    
    query = db.query(WeeklySheet).filter(WeeklySheet.assigned_to == current_user.id)
    
    # Filter by month if provided
    if month:
        query = query.filter(WeeklySheet.month == month)
    
    # Filter by year if provided  
    if year:
        query = query.filter(WeeklySheet.year == year)
    
    return query.order_by(WeeklySheet.created_at.desc()).all()

@app.get("/employees/weekly-sheets/current", response_model=Optional[WeeklySheetResponse])
def get_current_month_sheet(
    db: Session = Depends(get_db),
    current_user: User = Depends(check_employee_role)
):
    """Get the current month's weekly sheet for the employee"""
    
    current_month, current_year = get_current_month_year()
    
    sheet = db.query(WeeklySheet).filter(
        WeeklySheet.assigned_to == current_user.id,
        WeeklySheet.month == current_month,
        WeeklySheet.year == current_year
    ).first()
    
    return sheet

@app.post("/employees/weekly-sheets/{sheet_id}/copy", response_model=WeeklySheetResponse)
def copy_sheet_for_employee(
    sheet_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_employee_role)
):
    """Employee creates their own copy of an assigned sheet to work on"""
    
    # Find the original sheet
    original_sheet = db.query(WeeklySheet).filter(
        WeeklySheet.sheet_id == sheet_id,
        WeeklySheet.assigned_to == current_user.id
    ).first()
    
    if not original_sheet:
        raise HTTPException(status_code=404, detail="Sheet not found or not assigned to you")
    
    # Check if employee already has a working copy
    existing_copy = db.query(WeeklySheet).filter(
        WeeklySheet.month == original_sheet.month,
        WeeklySheet.year == original_sheet.year,
        WeeklySheet.created_by == current_user.id,
        WeeklySheet.assigned_to == current_user.id
    ).first()
    
    if existing_copy:
        raise HTTPException(status_code=400, detail="You already have a working copy for this month")
    
    # Create employee's working copy
    employee_sheet = WeeklySheet(
        sheet_id=generate_sheet_id(),
        month=original_sheet.month,
        year=original_sheet.year,
        created_by=current_user.id,
        assigned_to=current_user.id,
        is_template=False
    )
    
    db.add(employee_sheet)
    db.commit()
    db.refresh(employee_sheet)
    
    # Copy all entries
    for original_entry in original_sheet.entries:
        new_entry = WeeklySheetEntry(
            sheet_id=employee_sheet.id,
            client_name=original_entry.client_name,
            week_number=original_entry.week_number,
            posts_count=original_entry.posts_count,
            reels_count=original_entry.reels_count,
            story_description=original_entry.story_description,
            is_topical_day=original_entry.is_topical_day
        )
        db.add(new_entry)
    
    db.commit()
    db.refresh(employee_sheet)
    
    return employee_sheet

@app.put("/employees/weekly-sheets/{sheet_id}", response_model=WeeklySheetResponse)
def update_employee_sheet(
    sheet_id: str,
    updates: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(check_employee_role)
):
    """Employee updates their own weekly sheet"""
    
    sheet = db.query(WeeklySheet).filter(
        WeeklySheet.sheet_id == sheet_id,
        WeeklySheet.created_by == current_user.id
    ).first()
    
    if not sheet:
        raise HTTPException(status_code=404, detail="Sheet not found or not owned by you")
    
    if sheet.status == "submitted":
        raise HTTPException(status_code=400, detail="Cannot edit submitted sheet")
    
    # Update entries if provided
    if "entries" in updates:
        for entry_update in updates["entries"]:
            if "id" in entry_update:
                entry = db.query(WeeklySheetEntry).filter(
                    WeeklySheetEntry.id == entry_update["id"],
                    WeeklySheetEntry.sheet_id == sheet.id
                ).first()
                
                if entry:
                    if "posts_count" in entry_update:
                        entry.posts_count = entry_update["posts_count"]
                    if "reels_count" in entry_update:
                        entry.reels_count = entry_update["reels_count"]
                    if "story_description" in entry_update:
                        entry.story_description = entry_update["story_description"]
    
    sheet.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(sheet)
    
    return sheet

@app.post("/employees/weekly-sheets/{sheet_id}/submit", response_model=WeeklySheetResponse)
def submit_employee_sheet(
    sheet_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_employee_role)
):
    """Employee submits their weekly sheet for review"""
    
    sheet = db.query(WeeklySheet).filter(
        WeeklySheet.sheet_id == sheet_id,
        WeeklySheet.created_by == current_user.id
    ).first()
    
    if not sheet:
        raise HTTPException(status_code=404, detail="Sheet not found or not owned by you")
    
    if sheet.status == "submitted":
        raise HTTPException(status_code=400, detail="Sheet already submitted")
    
    sheet.status = "submitted"
    sheet.submitted_at = datetime.utcnow()
    db.commit()
    db.refresh(sheet)
    
    return sheet

@app.get("/weekly-sheets/template", response_model=Dict)
def get_sheet_template(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a template structure for creating weekly sheets"""
    
    template = {
        "clients": DEFAULT_CLIENTS,
        "weeks": [
            {"week_number": 1, "date_range": "1-7"},
            {"week_number": 2, "date_range": "8-14"},
            {"week_number": 3, "date_range": "15-21"},
            {"week_number": 4, "date_range": "22-28"},
            {"week_number": 5, "date_range": "29-30"}
        ],
        "content_types": ["POSTS", "REELS", "STORY"],
        "story_templates": [
            "COLLAGE + WTSAP STORY",
            "GRID POST STORY",
            "CUSTOM CONTENT"
        ]
    }
    
    return template

# Employee dashboard - get tasks summary
# Employee dashboard
@app.get("/employees/dashboard")
async def get_employee_dashboard(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != UserRole.EMPLOYEE:
        raise HTTPException(status_code=403, detail="Access denied")

    # Get current time with timezone info
    now = datetime.now(IST)
    today_start = datetime.combine(now.date(), time.min).replace(tzinfo=IST)
    week_start = (today_start - timedelta(days=today_start.weekday())).replace(tzinfo=IST)
    month_start = datetime.combine(now.date().replace(day=1), time.min).replace(tzinfo=IST)

    # Check active session
    active_session = db.query(WorkSession).filter(
        WorkSession.employee_id == current_user.id,
        WorkSession.clock_out == None
    ).first()

    # Time calculation function
    def calculate_minutes(sessions):
        return sum(
            (session.clock_out - session.clock_in).total_seconds() / 60
            for session in sessions
            if session.clock_out
        )

    # Today's work
    today_sessions = db.query(WorkSession).filter(
        WorkSession.employee_id == current_user.id,
        WorkSession.clock_in >= today_start,
        WorkSession.clock_out != None
    ).all()
    today_minutes = calculate_minutes(today_sessions)

    # Weekly work
    week_sessions = db.query(WorkSession).filter(
        WorkSession.employee_id == current_user.id,
        WorkSession.clock_in >= week_start,
        WorkSession.clock_out != None
    ).all()
    week_minutes = calculate_minutes(week_sessions)

    # Monthly work
    month_sessions = db.query(WorkSession).filter(
        WorkSession.employee_id == current_user.id,
        WorkSession.clock_in >= month_start,
        WorkSession.clock_out != None
    ).all()
    month_minutes = calculate_minutes(month_sessions)

    # Task statistics
    pending_tasks = db.query(Task).filter(
        Task.employee_id == current_user.id,
        Task.status == TaskStatus.PENDING
    ).count()

    completed_tasks = db.query(Task).filter(
        Task.employee_id == current_user.id,
        Task.status == TaskStatus.COMPLETED
    ).count()

    # Salary data
    salary = db.query(EmployeeSalary).filter(
        EmployeeSalary.employee_id == current_user.id
    ).order_by(EmployeeSalary.effective_from.desc()).first()

    # Initialize payroll period data
    payroll_period_data = None
    
    try:
        # Find current payroll period
        current_payroll_period = db.query(PayrollPeriod).filter(
            PayrollPeriod.start_date <= now.replace(tzinfo=None),  # Remove timezone for comparison
            PayrollPeriod.end_date >= now.replace(tzinfo=None)     # Remove timezone for comparison
        ).first()
        
        if current_payroll_period:
            # Calculate payroll period work
            payroll_period_sessions = db.query(WorkSession).filter(
                WorkSession.employee_id == current_user.id,
                WorkSession.clock_in >= current_payroll_period.start_date,
                WorkSession.clock_in <= current_payroll_period.end_date,
                WorkSession.clock_out != None
            ).all()
            payroll_period_minutes = calculate_minutes(payroll_period_sessions)
            
            # Make date objects timezone aware for calculations
            start_date_aware = current_payroll_period.start_date.replace(tzinfo=IST)
            end_date_aware = current_payroll_period.end_date.replace(tzinfo=IST)
            
            # Calculate period statistics
            days_in_period = (end_date_aware - start_date_aware).days + 1
            days_elapsed = (now - start_date_aware).days + 1
            days_remaining = max(0, days_in_period - days_elapsed)
            
            # Build payroll period data structure
            payroll_period_data = {
                "id": current_payroll_period.id,
                "name": current_payroll_period.name,
                "start_date": current_payroll_period.start_date.isoformat(),
                "end_date": current_payroll_period.end_date.isoformat(),
                "status": current_payroll_period.status,
                "minutes_worked": payroll_period_minutes,
                "hours_worked": round(payroll_period_minutes / 60, 2),
                "days_total": days_in_period,
                "days_elapsed": days_elapsed,
                "days_remaining": days_remaining,
                "completion_percentage": round((days_elapsed / days_in_period) * 100, 2) if days_in_period > 0 else 0
            }
    except Exception as e:
        print(f"Error processing payroll period: {str(e)}")
        # Don't let payroll period errors crash the entire dashboard

    return {
        "is_working": active_session is not None,
        "clocked_in_since": active_session.clock_in.isoformat() if active_session else None,
        "today_minutes_worked": today_minutes,
        "week_minutes_worked": week_minutes,
        "month_minutes_worked": month_minutes,
        "pending_tasks": pending_tasks,
        "completed_tasks": completed_tasks,
        "current_salary": {
            "monthly_salary": salary.monthly_salary,
            "currency": salary.currency,
            "effective_from": salary.effective_from.isoformat()
        } if salary else None,
        "current_time": now.isoformat(),
        "current_payroll_period": payroll_period_data
    }

# Allocator dashboard
# Replace the existing get_allocator_dashboard function with this fixed version
@app.get("/allocators/dashboard")
def get_allocator_dashboard(
    db: Session = Depends(get_db),
    current_user: User = Depends(check_allocator_role)
):
    # Use timezone-aware timestamps
    now = datetime.now(IST)
    today = now.date()
    
    # Get employee IDs that have been rejected
    rejected_employee_ids = db.query(ApprovalRequest.employee_id).filter(
        ApprovalRequest.status == "rejected"
    ).subquery()
    
    # Get all employees with approved status (excluding rejected ones)
    employees = db.query(User).filter(
        User.role == UserRole.EMPLOYEE,
        User.is_approved == True,
        User.is_active == True,
        ~User.id.in_(rejected_employee_ids)  # Exclude rejected employees
    ).all()
    
    print(f"Found {len(employees)} employees for dashboard")
    
    employee_work_data = []
    currently_working_count = 0
    total_today_minutes = 0
    
    for employee in employees:
        # Check current work session
        active_session = db.query(WorkSession).filter(
            WorkSession.employee_id == employee.id,
            WorkSession.clock_out == None
        ).first()
        
        is_working = active_session is not None
        current_duration = 0
        
        if is_working:
            currently_working_count += 1
            # Ensure timezone-aware comparison
            if active_session.clock_in.tzinfo is None:
                aware_clock_in = active_session.clock_in.replace(tzinfo=IST)
            else:
                aware_clock_in = active_session.clock_in
            
            current_duration = (now - aware_clock_in).total_seconds() / 60
        
        # Calculate today's work duration
        today_start = datetime.combine(today, datetime.min.time()).replace(tzinfo=IST)
        today_end = datetime.combine(today, datetime.max.time()).replace(tzinfo=IST)
        
        today_completed_sessions = db.query(WorkSession).filter(
            WorkSession.employee_id == employee.id,
            WorkSession.clock_in >= today_start,
            WorkSession.clock_in <= today_end,
            WorkSession.clock_out != None
        ).all()
        
        today_minutes = 0
        for session in today_completed_sessions:
            if session.duration_minutes is not None:
                today_minutes += session.duration_minutes
            else:
                # Calculate duration if not stored
                if session.clock_in and session.clock_out:
                    # Make both timezone aware
                    if session.clock_in.tzinfo is None:
                        in_time = session.clock_in.replace(tzinfo=IST)
                    else:
                        in_time = session.clock_in
                        
                    if session.clock_out.tzinfo is None:
                        out_time = session.clock_out.replace(tzinfo=IST)
                    else:
                        out_time = session.clock_out
                        
                    duration = (out_time - in_time).total_seconds() / 60
                    today_minutes += duration
        
        if is_working and active_session.clock_in.date() == today:
            today_minutes += current_duration
        
        total_today_minutes += today_minutes
        
        # Calculate weekly work duration
        week_start_date = today - timedelta(days=today.weekday())
        week_start = datetime.combine(week_start_date, datetime.min.time()).replace(tzinfo=IST)
        
        week_end_date = week_start_date + timedelta(days=6)
        week_end = datetime.combine(week_end_date, datetime.max.time()).replace(tzinfo=IST)
        
        week_completed_sessions = db.query(WorkSession).filter(
            WorkSession.employee_id == employee.id,
            WorkSession.clock_in >= week_start,
            WorkSession.clock_in <= week_end,
            WorkSession.clock_out != None
        ).all()
        
        week_minutes = 0
        for session in week_completed_sessions:
            if session.duration_minutes is not None:
                week_minutes += session.duration_minutes
            else:
                # Calculate duration if not stored
                if session.clock_in and session.clock_out:
                    # Make both timezone aware
                    if session.clock_in.tzinfo is None:
                        in_time = session.clock_in.replace(tzinfo=IST)
                    else:
                        in_time = session.clock_in
                        
                    if session.clock_out.tzinfo is None:
                        out_time = session.clock_out.replace(tzinfo=IST)
                    else:
                        out_time = session.clock_out
                        
                    duration = (out_time - in_time).total_seconds() / 60
                    week_minutes += duration
        
        if is_working and active_session.clock_in.date() >= week_start.date():
            week_minutes += current_duration
        
        # Get assigned tasks
        employee_tasks = db.query(Task).filter(Task.employee_id == employee.id).all()
        pending_tasks = sum(1 for task in employee_tasks if task.status == 'pending')
        completed_tasks = sum(1 for task in employee_tasks if task.status in ['completed', 'approved', 'rejected'])
        
        employee_work_data.append({
            "employee_id": employee.id,
            "employee_name": employee.username,
            "is_working": is_working,
            "clocked_in_since": active_session.clock_in if is_working else None,
            "current_session_minutes": round(current_duration, 2),
            "today_minutes_worked": round(today_minutes, 2),
            "today_hours_worked": round(today_minutes / 60, 2),
            "week_minutes_worked": round(week_minutes, 2),
            "week_hours_worked": round(week_minutes / 60, 2),
            "pending_tasks": pending_tasks,
            "completed_tasks": completed_tasks
        })
    
    # Sort by working status and time worked
    employee_work_data.sort(key=lambda x: (-x["is_working"], -x["today_minutes_worked"]))
    
    # Get task counts by client
    client_task_counts = {}
    clients = db.query(Client).all()
    
    for client in clients:
        count = db.query(Task).filter(
            Task.client_id == client.id,
            Task.allocator_id == current_user.id
        ).count()
        
        if count > 0:
            client_task_counts[client.name] = count
    
    # Get task statistics
    tasks = db.query(Task).filter(Task.allocator_id == current_user.id).all()
    
    pending_tasks = sum(1 for task in tasks if task.status == 'pending')
    completed_tasks = sum(1 for task in tasks if task.status == 'completed')
    approved_tasks = sum(1 for task in tasks if task.status == 'approved')
    rejected_tasks = sum(1 for task in tasks if task.status == 'rejected')
    completed_tasks_awaiting_review = sum(1 for task in tasks 
                                        if task.status == 'completed' and 
                                        (not task.report or not task.report.allocator_feedback))
    
    avg_minutes_today = 0
    if len(employees) > 0:
        avg_minutes_today = total_today_minutes / len(employees)
    
    return {
        "current_time_ist": now.isoformat(),
        "currently_working_employees": currently_working_count,
        "total_employees": len(employees),
        "total_today_minutes_worked": round(total_today_minutes, 2),
        "total_today_hours_worked": round(total_today_minutes / 60, 2),
        "average_work_minutes_today": round(avg_minutes_today, 2),
        "employee_work_data": employee_work_data,
        "client_task_counts": client_task_counts,
        "pending_tasks": pending_tasks,
        "completed_tasks": completed_tasks,
        "approved_tasks": approved_tasks,
        "rejected_tasks": rejected_tasks,
        "completed_tasks_awaiting_review": completed_tasks_awaiting_review
    }

@app.post("/employees/clock-in", response_model=WorkSessionResponse)
def employee_clock_in(
    session_data: WorkSessionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_employee_role)
):
    # Check if employee has an active session
    active_session = db.query(WorkSession).filter(
        WorkSession.employee_id == current_user.id,
        WorkSession.clock_out == None
    ).first()
    
    if active_session:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You already have an active work session"
        )
    
    # Create new work session with IST time
    new_session = WorkSession(
        employee_id=current_user.id,
        clock_in=datetime.now(IST),
        notes=session_data.notes
    )
    
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    
    return new_session

# Clock-out endpoint# Clock-out endpoint
@app.post("/employees/clock-out", response_model=WorkSessionResponse)
def employee_clock_out(
    session_data: WorkSessionUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_employee_role)
):
    # Find active session
    active_session = db.query(WorkSession).filter(
        WorkSession.employee_id == current_user.id,
        WorkSession.clock_out == None
    ).first()
    
    if not active_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active work session found"
        )
    
    # Update session with clock-out time in IST
    current_time = datetime.now(IST)
    active_session.clock_out = current_time
    
    # Calculate duration in minutes - handle timezone-aware comparison
    if active_session.clock_in.tzinfo is None:
        # If clock_in is naive, convert to aware with IST timezone
        aware_clock_in = active_session.clock_in.replace(tzinfo=IST)
    else:
        aware_clock_in = active_session.clock_in
        
    duration = (current_time - aware_clock_in).total_seconds() / 60
    active_session.duration_minutes = duration
    
    # Update notes if provided
    if session_data.notes:
        if active_session.notes:
            active_session.notes += f"\n{session_data.notes}"
        else:
            active_session.notes = session_data.notes
    
    db.commit()
    db.refresh(active_session)
    
    return active_session

# Get current work session status
# Get work session status
@app.get("/employees/work-session/status")
def get_work_session_status(
    db: Session = Depends(get_db),
    current_user: User = Depends(check_employee_role)
):
    active_session = db.query(WorkSession).filter(
        WorkSession.employee_id == current_user.id,
        WorkSession.clock_out == None
    ).first()
    
    if active_session:
        # Make sure active_session.clock_in has timezone info
        if active_session.clock_in.tzinfo is None:
            # If clock_in is naive, convert to aware with IST timezone
            aware_clock_in = active_session.clock_in.replace(tzinfo=IST)
        else:
            aware_clock_in = active_session.clock_in
            
        # Calculate duration with timezone-aware objects
        current_duration = (datetime.now(IST) - aware_clock_in).total_seconds() / 60
        
        return {
            "is_clocked_in": True,
            "session_id": active_session.id,
            "clock_in_time": active_session.clock_in,
            "current_duration_minutes": round(current_duration, 2)
        }
    
    return {
        "is_clocked_in": False,
        "session_id": None,
        "clock_in_time": None,
        "current_duration_minutes": 0
    }

@app.get("/employees/me/salary", response_model=EmployeeSalaryResponse)
def get_employee_salary_info(
    db: Session = Depends(get_db),
    current_user: User = Depends(check_employee_role)
):
    """Get current salary information for the logged-in employee"""
    salary = db.query(EmployeeSalary).filter(
        EmployeeSalary.employee_id == current_user.id
    ).order_by(EmployeeSalary.effective_from.desc()).first()
    
    if not salary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No salary information found"
        )
    
    return salary

# For allocators to view employee work sessions
@app.get("/allocators/employee/{employee_id}/work-sessions", response_model=List[WorkSessionResponse])
def get_employee_work_sessions_for_allocator(
    employee_id: int,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_allocator_role)
):
    # Check if employee exists
    employee = get_user_by_id(db, employee_id)
    if not employee or employee.role != UserRole.EMPLOYEE:
        raise HTTPException(status_code=404, detail="Employee not found")
    
    query = db.query(WorkSession).filter(WorkSession.employee_id == employee_id)
    
    # Filter by date range if provided
    if start_date:
        start_datetime = datetime.combine(start_date, datetime.min.time())
        query = query.filter(WorkSession.clock_in >= start_datetime)
    
    if end_date:
        end_datetime = datetime.combine(end_date, datetime.max.time())
        query = query.filter(WorkSession.clock_in <= end_datetime)
    
    # Order by clock_in time (newest first)
    query = query.order_by(WorkSession.clock_in.desc())
    
    return query.all()

@app.post("/payroll/periods", response_model=PayrollPeriodResponse)
def create_payroll_period(
    period_data: PayrollPeriodCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_allocator_role)
):
    """Create a new payroll period"""
    try:
        period = PayrollPeriod(
            name=period_data.name,
            start_date=period_data.start_date,
            end_date=period_data.end_date,
            created_by=current_user.id
        )
        db.add(period)
        db.commit()
        db.refresh(period)
        return period
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error creating payroll period: {str(e)}"
        )

@app.post("/payroll/periods/{period_id}/calculate", response_model=List[PayrollRecordResponse])
def calculate_payroll(
    period_id: int,
    params: PayrollCalculationParams,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_allocator_role)
):
    """Calculate payroll for all employees in a period"""
    try:
        records = generate_payroll_for_period(db, period_id, params)
        return records
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@app.post("/payroll/periods/{period_id}/lock")
def lock_payroll_period(
    period_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_allocator_role)
):
    """Lock a payroll period to prevent changes"""
    period = db.query(PayrollPeriod).get(period_id)
    if not period:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Payroll period not found"
        )
    
    if period.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only completed periods can be locked"
        )
    
    period.status = "locked"
    period.locked_at = datetime.utcnow()
    db.commit()
    
    return {"message": "Payroll period locked successfully"}

@app.get("/payroll/periods", response_model=List[PayrollPeriodResponse])
def get_payroll_periods(
    db: Session = Depends(get_db),
    current_user: User = Depends(check_allocator_role)
):
    """Get all payroll periods"""
    return db.query(PayrollPeriod).order_by(PayrollPeriod.created_at.desc()).all()

@app.get("/payroll/periods/{period_id}", response_model=List[PayrollRecordResponse])
def get_payroll_period_details(
    period_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_allocator_role)
):
    """Get all payroll records for a period"""
    period = db.query(PayrollPeriod).get(period_id)
    if not period:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Payroll period not found"
        )
    
    return db.query(PayrollRecord).filter(
        PayrollRecord.period_id == period_id
    ).all()

@app.get("/tasks/employee/{employee_id}", response_model=List[TaskResponse])
def get_tasks_by_employee(
    employee_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get tasks assigned to a specific employee"""
    # For employees, they can only see their own tasks
    if current_user.role == UserRole.EMPLOYEE and current_user.id != employee_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view tasks for other employees"
        )
    
    # For allocators, check if the employee exists
    if current_user.role == UserRole.ALLOCATOR:
        employee = db.query(User).filter(
            User.id == employee_id,
            User.role == UserRole.EMPLOYEE
        ).first()
        
        if not employee:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Employee not found"
            )
    
    # Get tasks for this employee
    tasks = db.query(Task).filter(Task.employee_id == employee_id).all()
    
    return tasks

@app.get("/employees/payroll", response_model=List[PayrollRecordResponse])
def get_employee_payroll(
    db: Session = Depends(get_db),
    current_user: User = Depends(check_employee_role)
):
    """Get payroll records for the current employee with period details"""
    records = db.query(PayrollRecord).options(
        joinedload(PayrollRecord.period)  # Explicitly load period
    ).filter(
        PayrollRecord.employee_id == current_user.id
    ).order_by(PayrollRecord.created_at.desc()).all()
    
    return records

@app.get("/employees/payroll/{period_id}", response_model=PayrollRecordResponse)
def get_employee_payroll_detail(
    period_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_employee_role)
):
    """Get a specific payroll record for the current employee"""
    record = db.query(PayrollRecord).filter(
        PayrollRecord.employee_id == current_user.id,
        PayrollRecord.period_id == period_id
    ).first()
    
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Payroll record not found"
        )
    
    return record

@app.get("/employees/net-salary", response_model=Dict)
def get_employee_net_salary(
    db: Session = Depends(get_db),
    current_user: User = Depends(check_employee_role)
):
    """Get current month's net salary calculation with deductions for the logged-in employee"""
    # Get current salary
    salary = get_employee_salary(db, current_user.id)
    if not salary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No salary information found"
        )
    
    # Get current month's date range
    now = datetime.now(IST)
    month_start = datetime.combine(now.date().replace(day=1), time.min).replace(tzinfo=IST)
    next_month = now.replace(day=28) + timedelta(days=4)  # Move to next month
    month_end = datetime.combine(next_month.replace(day=1) - timedelta(days=1), time.max).replace(tzinfo=IST)
    
    # Get all completed work sessions in current month
    sessions = db.query(WorkSession).filter(
        WorkSession.employee_id == current_user.id,
        WorkSession.clock_in >= month_start,
        WorkSession.clock_out <= month_end,
        WorkSession.clock_out.isnot(None)
    ).all()
    
    # Calculate total minutes worked
    total_minutes = sum(
        (session.clock_out - session.clock_in).total_seconds() / 60
        for session in sessions
    )
    
    # Standard work minutes for the month
    standard_hours = 160
    standard_minutes = standard_hours * 60
    
    # Calculate overtime and undertime
    overtime_minutes = max(0, total_minutes - standard_minutes)
    undertime_minutes = max(0, standard_minutes - total_minutes)
    
    # Calculate rates
    hourly_rate = salary.monthly_salary / standard_hours
    overtime_rate = hourly_rate * 1.5
    
    # Calculate earnings and deductions
    base_salary = salary.monthly_salary
    overtime_pay = (overtime_minutes / 60) * overtime_rate
    undertime_deduction = (undertime_minutes / 60) * hourly_rate
    
    # Calculate net salary
    net_salary = base_salary + overtime_pay - undertime_deduction
    
    # Calculate percentage of standard time worked
    percent_worked = (total_minutes / standard_minutes) * 100 if standard_minutes > 0 else 0
    
    return {
        "employee_id": current_user.id,
        "month": now.strftime("%B %Y"),
        "base_salary": base_salary,
        "hourly_rate": hourly_rate,
        "currency": salary.currency,
        
        "standard_hours": standard_hours,
        "hours_worked": total_minutes / 60,
        "percent_standard_time_worked": round(percent_worked, 2),
        
        "overtime_hours": overtime_minutes / 60,
        "overtime_rate": overtime_rate,
        "overtime_pay": overtime_pay,
        
        "undertime_hours": undertime_minutes / 60,
        "undertime_deduction": undertime_deduction,
        
        "net_salary": net_salary,
        
        "period_start": month_start,
        "period_end": month_end,
        "last_updated": now
    }
@app.post("/clients/add", response_model=ClientResponse)
async def add_client(
    client_data: ClientCreate,
    current_user: User = Depends(check_allocator_role),
    db: Session = Depends(get_db)
):
    """
    Add a new client (Allocator only)
    """
    # Check if client already exists
    existing_client = db.query(Client).filter(Client.name == client_data.name).first()
    if existing_client:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Client '{client_data.name}' already exists"
        )
    
    # Create new client
    new_client = Client(name=client_data.name)
    db.add(new_client)
    db.commit()
    db.refresh(new_client)
    
    return new_client

@app.delete("/clients/delete/{client_id}")
async def delete_client(
    client_id: int,
    current_user: User = Depends(check_allocator_role),
    db: Session = Depends(get_db)
):
    """
    Delete a client (Allocator only)
    """
    # Get the client
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Client not found"
        )
    
    # Check if client has associated tasks
    task_count = db.query(Task).filter(Task.client_id == client_id).count()
    if task_count > 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot delete client '{client.name}' because it has {task_count} associated tasks"
        )
    
    # Check if client has associated weekly sheet entries
    sheet_entries_count = db.query(WeeklySheetEntry).filter(WeeklySheetEntry.client_name == client.name).count()
    if sheet_entries_count > 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot delete client '{client.name}' because it has weekly sheet entries"
        )
    
    # Delete the client
    db.delete(client)
    db.commit()
    
    return {"message": f"Client '{client.name}' deleted successfully"}
@app.get("/employees/timesheets", response_model=List[Dict])
def get_employee_timesheets(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_employee_role)
):
    """Get timesheet data (work sessions) for the current employee"""
    query = db.query(WorkSession).filter(
        WorkSession.employee_id == current_user.id,
        WorkSession.clock_out.isnot(None)  # Only completed sessions
    )
    
    # Filter by date range if provided
    if start_date:
        start_datetime = datetime.combine(start_date, datetime.min.time())
        query = query.filter(WorkSession.clock_in >= start_datetime)
    
    if end_date:
        end_datetime = datetime.combine(end_date, datetime.max.time())
        query = query.filter(WorkSession.clock_in <= end_datetime)
    
    # Order by clock_in time (newest first)
    sessions = query.order_by(WorkSession.clock_in.desc()).all()
    
    # Group by date
    timesheets = {}
    for session in sessions:
        # Get the date in string format
        date_str = session.clock_in.date().isoformat()
        
        if date_str not in timesheets:
            timesheets[date_str] = {
                "date": date_str,
                "total_minutes": 0,
                "sessions": []
            }
        
        # Add session details
        session_data = {
            "id": session.id,
            "clock_in": session.clock_in,
            "clock_out": session.clock_out,
            "duration_minutes": session.duration_minutes or 0,
            "notes": session.notes
        }
        
        timesheets[date_str]["sessions"].append(session_data)
        timesheets[date_str]["total_minutes"] += session.duration_minutes or 0
    
    # Convert to list sorted by date (newest first)
    result = list(timesheets.values())
    result.sort(key=lambda x: x["date"], reverse=True)
    
    return result    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
#new endpoint added