import logging
import os
import secrets
from functools import lru_cache
from fastapi import FastAPI, HTTPException, Query, Request, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import pandas as pd
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create limiter before app
limiter = Limiter(key_func=get_remote_address)

# FastAPI App - Define this once at the beginning
app = FastAPI(
    title="Cricket Match Analysis API",
    description="Provides detailed analysis of cricket matches between two teams",
    version="1.2.0"
)

# Set up limiter with the app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Secure API Key Management
# DEVELOPMENT API KEY - Replace with secure method in production
DEV_API_KEY = "cricket-dev-api-key-123"
SECURE_API_KEY = os.getenv("CRICKET_API_KEY", DEV_API_KEY)
VALID_API_KEYS = {SECURE_API_KEY, DEV_API_KEY}
logger.info(f"Development API Key: {DEV_API_KEY} (use this for testing)")

# API Key Authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: Optional[str] = Security(api_key_header)):
    if api_key not in VALID_API_KEYS:
        logger.warning("Unauthorized API access attempt")
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key

# Pydantic Models
class MatchDetails(BaseModel):
    date: Optional[str] = None
    venue: Optional[str] = None
    winner: Optional[str] = None
    winner_runs: Optional[float] = None
    winner_wickets: Optional[float] = None
    method: Optional[str] = None

class MatchSummary(BaseModel):
    total_matches: int
    team1_wins: int
    team2_wins: int
    tied_matches: int
    team1_avg_runs: float
    team2_avg_runs: float
    team1_avg_wickets: float
    team2_avg_wickets: float
    matches_details: List[MatchDetails]

# Data Analyzer
class CricketDataAnalyzer:
    def __init__(self, csv_path: str):
        try:
            # Try to read the CSV file
            if os.path.exists(csv_path):
                self.df = pd.read_csv(csv_path)
                logger.info(f"Loaded {len(self.df)} matches successfully")
            else:
                logger.error(f"CSV file not found: {csv_path}")
                self.df = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            self.df = pd.DataFrame()

    def get_team_matches(self, team1: str, team2: str) -> Optional[MatchSummary]:
        if self.df.empty:
            logger.error("Dataframe is empty")
            return None

        # Clean input strings to remove trailing spaces
        team1 = team1.strip()
        team2 = team2.strip()

        try:
            team_matches = self.df[
                ((self.df['Team 1'].str.lower() == team1.lower()) & 
                 (self.df['Team 2'].str.lower() == team2.lower())) | 
                ((self.df['Team 1'].str.lower() == team2.lower()) & 
                 (self.df['Team 2'].str.lower() == team1.lower()))
            ]

            if team_matches.empty:
                logger.info(f"No matches found between {team1} and {team2}")
                return None

            team1_wins = len(team_matches[team_matches['Winner'].str.lower() == team1.lower()])
            team2_wins = len(team_matches[team_matches['Winner'].str.lower() == team2.lower()])
            tied_matches = len(team_matches[team_matches['Outcome'].str.lower().fillna('') == 'tie'])

            matches_details = [
                MatchDetails(
                    date=str(match['Date']) if pd.notna(match['Date']) else None,
                    venue=str(match['Venue']) if pd.notna(match['Venue']) else None,
                    winner=str(match['Winner']) if pd.notna(match['Winner']) else None,
                    winner_runs=float(match['Winner Runs']) if pd.notna(match['Winner Runs']) else None,
                    winner_wickets=float(match['Winner Wickets']) if pd.notna(match['Winner Wickets']) else None,
                    method=str(match['Method']) if pd.notna(match['Method']) else None
                ) for _, match in team_matches.iterrows()
            ]

            return MatchSummary(
                total_matches=len(team_matches),
                team1_wins=team1_wins,
                team2_wins=team2_wins,
                tied_matches=tied_matches,
                team1_avg_runs=round(team_matches['Team 1 Runs'].fillna(0).mean(), 2),
                team2_avg_runs=round(team_matches['Team 2 Runs'].fillna(0).mean(), 2),
                team1_avg_wickets=round(team_matches['Team 1 Wickets'].fillna(0).mean(), 2),
                team2_avg_wickets=round(team_matches['Team 2 Wickets'].fillna(0).mean(), 2),
                matches_details=matches_details
            )
        except Exception as e:
            logger.error(f"Error processing match data: {e}")
            return None

# Path to CSV file
CSV_PATH = os.getenv("CRICKET_CSV_PATH", "/home/moni/cricket analysis/info.csv")

# Instantiate Analyzer
analyzer = CricketDataAnalyzer(CSV_PATH)

@lru_cache(maxsize=128)
def cached_team_matches(team1: str, team2: str) -> Optional[MatchSummary]:
    """Cache results for better performance"""
    return analyzer.get_team_matches(team1, team2)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error in {request.url}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

# Routes
@app.get("/")
async def root():
    """Root endpoint that provides API information and key for development"""
    return {
        "api": "Cricket Match Analysis API",
        "version": "1.2.0",
        "dev_api_key": DEV_API_KEY,  # Only for development
        "endpoints": [
            {"path": "/teams", "description": "Get all available teams"},
            {"path": "/summary", "description": "Get match summary between two teams"},
            {"path": "/clear_cache", "description": "Clear data cache and reload"}
        ],
        "authentication": "Add this header to all requests: X-API-Key: " + DEV_API_KEY
    }

@app.get("/teams", dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")
async def get_available_teams(request: Request):
    """Get a list of all available cricket teams"""
    if analyzer.df.empty:
        return {"teams": [], "error": "No data available"}
        
    teams = sorted(set(analyzer.df['Team 1'].dropna().tolist() + analyzer.df['Team 2'].dropna().tolist()))
    return {"teams": teams}

@app.get("/summary", response_model=MatchSummary, dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")
async def get_match_summary(request: Request, team1: str, team2: str):
    """
    Get a summary of cricket matches between two teams
    - team1: Name of the first team
    - team2: Name of the second team
    """
    # Validate and clean input
    team1 = team1.strip()
    team2 = team2.strip()
    
    if team1.lower() == team2.lower():
        raise HTTPException(status_code=400, detail="Teams must be different")
    
    summary = cached_team_matches(team1, team2)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"No matches found between {team1} and {team2}")
    return summary

@app.get("/clear_cache", dependencies=[Depends(verify_api_key)])
@limiter.limit("2/minute")
async def clear_cache(request: Request):
    """Clear the cache and reload the cricket data"""
    cached_team_matches.cache_clear()
    global analyzer
    analyzer = CricketDataAnalyzer(CSV_PATH)
    logger.info("Cache cleared and data reloaded")
    return {"message": "Cache cleared and data reloaded"}

if __name__ == "__main__":
    # Use environment variables with defaults for host and port
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    logger.info(f"Starting Cricket API server on {host}:{port}")
    logger.info(f"Development API Key: {DEV_API_KEY}")
    uvicorn.run(app, host=host, port=port)