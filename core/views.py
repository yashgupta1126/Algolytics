import requests
import time
import numpy as np
import google.generativeai as genai
import markdown
from datetime import datetime, timezone
from sklearn.linear_model import LinearRegression
from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import CustomRegistrationForm, ProfileUpdateForm, CodeReviewForm, ComparisonForm

# --- 1. UPDATE HOME VIEW (Time Left Logic) ---
def home_view(request):
    if request.method == 'POST':
        u = request.POST.get('username')
        p = request.POST.get('password')
        user = authenticate(request, username=u, password=p)
        if user is not None:
            login(request, user)
            return redirect('dashboard')
        else:
            messages.error(request, "Invalid credentials.")
            return redirect('home')

    api_url = "https://contest-hive.vercel.app/api/all" 
    upcoming_contests = []
    try:
        response = requests.get(api_url, timeout=5)
        if response.status_code == 200:
            data = response.json().get('data', {})
            all_contests = []
            now = datetime.now(timezone.utc)
            
            for platform_contests in data.values():
                for contest in platform_contests:
                    # Calculate duration
                    sec = contest.get('duration', 0)
                    contest['duration_formatted'] = f"{sec // 3600}h {(sec % 3600) // 60}m"
                    
                    # Calculate Time Left
                    try:
                        start_time_str = contest.get('startTime', '').replace('Z', '+0000')
                        start_dt = datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M:%S%z")
                        delta = start_dt - now
                        if delta.total_seconds() > 0:
                            days = delta.days
                            hours = delta.seconds // 3600
                            if days > 0:
                                contest['time_left'] = f"In {days}d {hours}h"
                            else:
                                contest['time_left'] = f"In {hours}h { (delta.seconds % 3600) // 60 }m"
                        else:
                            contest['time_left'] = "Started"
                    except:
                        contest['time_left'] = "TBA"
                        
                all_contests.extend(platform_contests)
            
            all_contests.sort(key=lambda x: x.get('startTime', ''))
            upcoming_contests = all_contests[:6] 
    except requests.RequestException:
        pass

    return render(request, 'core/home.html', {'contests': upcoming_contests})

# --- 2. NEW DASHBOARD VIEW ---
@login_required(login_url='home')
def dashboard_view(request):
    return render(request, 'core/dashboard.html')

# --- 3. UPGRADED PROFILE VIEW (ML + Analytics) ---
@login_required(login_url='home')
def profile_view(request):
    handle = request.user.codeforces_handle
    profile_data = None
    error_message = None

    if handle:
        try:
            # Fetch Rating History
            rating_resp = requests.get(f"https://codeforces.com/api/user.rating?handle={handle}", timeout=5).json()
            time.sleep(0.5) # Polite delay
            # Fetch Submission History
            status_resp = requests.get(f"https://codeforces.com/api/user.status?handle={handle}", timeout=8).json()

            if rating_resp.get('status') == 'OK' and status_resp.get('status') == 'OK':
                contests = rating_resp['result']
                submissions = status_resp['result']

                # --- 1. Machine Learning Prediction Logic ---
                labels, y_list, trend_line, future_preds = [], [], [], []
                current_rating, next_predicted = 0, 0
                
                if len(contests) >= 3:
                    X = np.array([i+1 for i in range(len(contests))]).reshape(-1, 1)
                    y = np.array([c['newRating'] for c in contests])
                    labels = [f"C{i+1}" for i in range(len(contests))]
                    
                    overall_model = LinearRegression().fit(X, y)
                    trend_line = overall_model.predict(X).astype(int).tolist()
                    
                    recent_window = min(len(contests), 15)
                    recent_model = LinearRegression().fit(X[-recent_window:], y[-recent_window:])
                    
                    future_X = np.array([len(contests) + 1, len(contests) + 2]).reshape(-1, 1)
                    future_preds = recent_model.predict(future_X).astype(int).tolist()
                    
                    labels.extend(["P1", "P2"])
                    y_list = y.tolist()
                    current_rating = int(y[-1])
                    next_predicted = int(future_preds[0])

                # --- 2. Problem Ratings Logic (Histogram) ---
                accepted = [s for s in submissions if s.get('verdict') == 'OK']
                # Get unique solved problems to avoid counting the same problem twice
                unique_solved = {s['problem']['name']: s['problem'] for s in accepted}.values()
                
                rating_counts = {}
                tag_counts = {}
                
                for prob in unique_solved:
                    # Count Ratings (e.g., 800, 900, 1000)
                    if 'rating' in prob:
                        r = prob['rating']
                        rating_counts[r] = rating_counts.get(r, 0) + 1
                    
                    # Count Tags (e.g., math, greedy)
                    for tag in prob.get('tags', []):
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1

                # Sort Histogram data by rating ascending
                sorted_ratings = sorted(rating_counts.items())
                hist_labels = [str(r[0]) for r in sorted_ratings]
                hist_data = [r[1] for r in sorted_ratings]

                # Sort Pie chart data by frequency descending
                sorted_tags = sorted(tag_counts.items(), key=lambda item: item[1], reverse=True)
                pie_labels = [t[0] for t in sorted_tags]
                pie_data = [t[1] for t in sorted_tags]

                profile_data = {
                    'handle': handle,
                    # ML Data
                    'ml_labels': labels,
                    'ml_actual': y_list,
                    'ml_trend': trend_line,
                    'ml_future': future_preds,
                    'current_rating': current_rating,
                    'next_predicted': next_predicted,
                    # Histogram Data
                    'hist_labels': hist_labels,
                    'hist_data': hist_data,
                    # Pie Chart Data
                    'pie_labels': pie_labels,
                    'pie_data': pie_data,
                    'total_solved': len(unique_solved)
                }
            else:
                error_message = "Failed to fetch data from Codeforces."
        except Exception as e:
            error_message = "Could not load analytics. Please check your network or try again."

    return render(request, 'core/profile.html', {'profile_data': profile_data, 'error_message': error_message})

# 4. The Logout View
def logout_view(request):
    logout(request)
    return redirect('home')

# 1. The Registration View
def register_view(request):
    # If they are already logged in, send them to the dashboard
    if request.user.is_authenticated:
        return redirect('profile')
        
    if request.method == 'POST':
        form = CustomRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user) # Automatically log them in after registering
            messages.success(request, f"Account created successfully for {user.username}!")
            return redirect('profile')
    else:
        form = CustomRegistrationForm()
        
    return render(request, 'core/register.html', {'form': form})

# 2. The Profile Update View
@login_required(login_url='home')
def update_profile_view(request):
    if request.method == 'POST':
        # request.user pre-fills the form with their current data
        form = ProfileUpdateForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, "Profile updated successfully!")
            return redirect('profile')
    else:
        form = ProfileUpdateForm(instance=request.user)
        
    return render(request, 'core/update_profile.html', {'form': form})

@login_required(login_url='home')
def ai_code_review_view(request):
    ai_response = None
    
    if request.method == 'POST':
        form = CodeReviewForm(request.POST)
        if form.is_valid():
            problem_link = form.cleaned_data['problem_link']
            user_code = form.cleaned_data['code']
            
            genai.configure(api_key=settings.GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-2.5-flash') 
            
            # THE BULLETPROOF FORMATTING PROMPT
            prompt = f"""
            You are an Expert Competitive Programming Coach. 
            Analyze this code for problem: {problem_link}
            
            Code:
            ```
            {user_code}
            ```
            
            Provide a highly detailed, professional review formatted strictly in Markdown. Follow these formatting rules EXACTLY:
            - NEVER use backslashes to escape underscores. Do NOT write `S\_i`.
            - Wrap ALL code variables, array names, and function names in standard Markdown backticks (e.g., `S_i`, `helper()`, `dp[i][odd]`).
            - Use single dollar signs ($) ONLY for Big-O notation or pure mathematical formulas (e.g., $O(N \log N)$, $O(T \cdot N)$, $2 \cdot 10^5$). DO NOT use dollar signs for variable names.
            
            Format your response exactly using these sections:
            
            ### 1. Time & Space Complexity
            Provide a detailed Big-O analysis. Break down the cost of loops, data structures, and the total complexity per test case.
            
            ### 2. Logical Flaws & Edge Cases
            Identify any logical misunderstandings. You MUST provide specific, concrete examples of inputs where the code fails. 
            Format examples clearly showing:
            * **Input:** * **Your Output:**
            * **Correct Output:**
            
            ### 3. Optimization & Fixes
            Explain exactly how to correct the logic or optimize the algorithm.
            
            CRITICAL: Do NOT write introductory or concluding filler. Output ONLY the requested Markdown analysis.
            """
            
            try:
                response = model.generate_content(prompt)
                ai_response = markdown.markdown(response.text, extensions=['fenced_code', 'tables'])
                
            except Exception as e:
                # GRACEFUL ERROR HANDLER: Catch rate limits without breaking the site
                error_msg = str(e).lower()
                if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
                    ai_response = """
                    <div class='alert alert-warning text-center mt-4'>
                        <h4 class='text-warning'>⏳ Too Many Requests</h4>
                        <p>Our AI is currently analyzing a high volume of code. To ensure fairness, we limit rapid successive requests.</p>
                        <p><strong>Please wait 60 seconds before trying again.</strong></p>
                    </div>
                    """
                else:
                    ai_response = f"<div class='alert alert-danger'>Error connecting to AI: {str(e)}</div>"
    else:
        form = CodeReviewForm()
        
    return render(request, 'core/code_review.html', {'form': form, 'ai_response': ai_response})

@login_required(login_url='home')
def compare_view(request):
    chart_data = None
    ai_analysis = None
    error_message = None
    
    initial_data = {}
    if request.user.codeforces_handle:
        initial_data['handle_1'] = request.user.codeforces_handle

    if request.method == 'POST':
        form = ComparisonForm(request.POST)
        if form.is_valid():
            h1 = form.cleaned_data['handle_1']
            h2 = form.cleaned_data['handle_2']

            try:
                # 1. Fetch Basic Info
                info_resp = requests.get(f"https://codeforces.com/api/user.info?handles={h1};{h2}", timeout=5).json()
                
                # 2. Add delays to prevent Codeforces from blocking our IP (Rate Limiting)
                time.sleep(0.5)
                s1_resp = requests.get(f"https://codeforces.com/api/user.status?handle={h1}", timeout=8).json()
                
                time.sleep(0.5)
                s2_resp = requests.get(f"https://codeforces.com/api/user.status?handle={h2}", timeout=8).json()

                # 3. Only proceed if ALL THREE API calls succeeded
                if info_resp.get('status') == 'OK' and s1_resp.get('status') == 'OK' and s2_resp.get('status') == 'OK':
                    u1_info, u2_info = info_resp['result'][0], info_resp['result'][1]
                    s1_data, s2_data = s1_resp['result'], s2_resp['result']

                    # --- Custom Analytics Engine ---
                    def analyze_submissions(subs):
                        total = len(subs)
                        if total == 0: return 0, 0, {}
                        
                        accepted = [s for s in subs if s.get('verdict') == 'OK']
                        # Calculate unique solved problems
                        unique_solved = len(set([s['problem']['name'] for s in accepted if 'name' in s['problem']]))
                        # Calculate acceptance rate mathematically
                        acc_rate = round((len(accepted) / total) * 100, 1)
                        
                        # Tally up their strongest problem tags
                        tag_counts = {}
                        for s in accepted:
                            for tag in s['problem'].get('tags', []):
                                tag_counts[tag] = tag_counts.get(tag, 0) + 1
                        return unique_solved, acc_rate, tag_counts

                    u1_solved, u1_acc, u1_tags = analyze_submissions(s1_data)
                    u2_solved, u2_acc, u2_tags = analyze_submissions(s2_data)

                    # Find the top 5 overlapping categories to compare
                    common_tags = list(set(list(u1_tags.keys()) + list(u2_tags.keys())))
                    common_tags.sort(key=lambda t: u1_tags.get(t, 0) + u2_tags.get(t, 0), reverse=True)
                    top_5_categories = common_tags[:5] if common_tags else ["Math", "Greedy", "DP", "Graphs", "Strings"]

                    u1_category_stats = [u1_tags.get(tag, 0) for tag in top_5_categories]
                    u2_category_stats = [u2_tags.get(tag, 0) for tag in top_5_categories]

                    # 4. Structure the exact JSON needed for the frontend Pie and Bar charts
                    chart_data = {
                        'u1_name': u1_info.get('handle', h1),
                        'u2_name': u2_info.get('handle', h2),
                        'solved': [u1_solved, u2_solved],
                        'acc_rates': [u1_acc, u2_acc],
                        'categories': top_5_categories,
                        'u1_categories': u1_category_stats,
                        'u2_categories': u2_category_stats
                    }

                    # 5. Generate AI Tactical Breakdown
                    genai.configure(api_key=settings.GEMINI_API_KEY)
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    
                    prompt = f"""
                    You are an expert Competitive Programming Coach. Analyze this head-to-head matchup:
                    
                    Player 1: {u1_info.get('handle')} (Rating: {u1_info.get('rating', 'Unrated')}, Solved: {u1_solved}, Acc Rate: {u1_acc}%)
                    Player 2: {u2_info.get('handle')} (Rating: {u2_info.get('rating', 'Unrated')}, Solved: {u2_solved}, Acc Rate: {u2_acc}%)
                    
                    Write a concise, professional comparison formatted in Markdown.
                    1. **The Matchup:** A 2-sentence summary.
                    2. **Strengths:** Who is more accurate? Who grinds more problems? 
                    3. **Improvement Focus:** Based on their stats, what should Player 1 focus on to beat Player 2?
                    
                    CRITICAL: Be extremely concise. No intro/outro fluff.
                    """
                    
                    try:
                        ai_resp = model.generate_content(prompt)
                        ai_analysis = markdown.markdown(ai_resp.text)
                    except Exception as e:
                        if "429" in str(e).lower() or "quota" in str(e).lower():
                            ai_analysis = "<div class='alert alert-warning'>AI is cooling down (Rate Limit). Please wait 60 seconds and try again.</div>"
                        else:
                            ai_analysis = f"<div class='alert alert-danger'>AI Error: {str(e)}</div>"

                else:
                    error_message = "Codeforces API error. One or both handles might not exist, or Codeforces blocked our request."
                    
            except Exception as e:
                error_message = f"Backend Network Error: {str(e)}"
    else:
        form = ComparisonForm(initial=initial_data)

    context = {
        'form': form,
        'chart_data': chart_data,
        'ai_analysis': ai_analysis,
        'error_message': error_message
    }
    return render(request, 'core/compare.html', context)


@login_required(login_url='home')
def predict_rating_view(request):
    chart_data = None
    error_message = None
    
    # Check if the user has linked their Codeforces handle in their profile
    handle = request.user.codeforces_handle
    if not handle:
        error_message = "Please update your profile with your Codeforces handle to use the prediction engine."
        return render(request, 'core/predict.html', {'error_message': error_message})

    try:
        # 1. Fetch Historical Rating Data
        resp = requests.get(f"https://codeforces.com/api/user.rating?handle={handle}", timeout=5).json()
        
        if resp['status'] == 'OK' and len(resp['result']) >= 3:
            contests = resp['result']
            
            X = np.array([i+1 for i in range(len(contests))]).reshape(-1, 1)
            y = np.array([c['newRating'] for c in contests])
            labels = [f"Contest {i+1}" for i in range(len(contests))]
            
            # --- UPGRADED ML LOGIC ---
            
            # 2. Overall Trendline (Degree 1 Linear Regression)
            # This draws a clean, straight baseline across your entire history
            overall_model = LinearRegression()
            overall_model.fit(X, y)
            historical_trend = overall_model.predict(X).astype(int).tolist()
            
            # 3. Future Prediction (Recent Form Momentum)
            # We train a second model ONLY on your last 15 contests to predict your immediate future
            recent_window = min(len(contests), 15)
            X_recent = X[-recent_window:]
            y_recent = y[-recent_window:]
            
            recent_model = LinearRegression()
            recent_model.fit(X_recent, y_recent)
            
            future_X = np.array([len(contests) + 1, len(contests) + 2, len(contests) + 3]).reshape(-1, 1)
            future_predictions = recent_model.predict(future_X).astype(int).tolist()
            
            # Extend the labels for the future contests
            labels.extend(["Predicted 1", "Predicted 2", "Predicted 3"])
            
            # Structure the JSON for the frontend Line Chart
            chart_data = {
                'handle': handle,
                'labels': labels,
                'actual_ratings': y.tolist(),
                'trend_line': historical_trend,
                'predictions': future_predictions,
                'current_rating': int(y[-1]),
                'next_predicted': int(future_predictions[0])
            }
        else:
            error_message = "Not enough contest history. You need at least 3 rated contests to generate a prediction."
            
    except Exception as e:
        error_message = f"Failed to fetch data or generate prediction: {str(e)}"

    return render(request, 'core/predict.html', {
        'chart_data': chart_data, 
        'error_message': error_message
    })


# filepath: core/views.py

@login_required(login_url='home')
def weak_spot_view(request):
    ai_roadmap = None
    error_message = None
    
    handle = request.user.codeforces_handle
    if not handle:
        error_message = "Please update your profile with your Codeforces handle to use the Weak-Spot Engine."
        return render(request, 'core/weak_spot.html', {'error_message': error_message})

    if request.method == 'POST':
        try:
            # 1. Fetch the last 100 submissions
            url = f"https://codeforces.com/api/user.status?handle={handle}&from=1&count=100"
            resp = requests.get(url, timeout=8).json()

            if resp.get('status') == 'OK':
                submissions = resp['result']
                
                # 2. Filter for failures (TLE, WA, MLE, RE)
                failed_verdicts = ['TIME_LIMIT_EXCEEDED', 'WRONG_ANSWER', 'MEMORY_LIMIT_EXCEEDED', 'RUNTIME_ERROR']
                failed_subs = [s for s in submissions if s.get('verdict') in failed_verdicts]
                
                # 3. Extract and count the algorithmic tags from those failed problems
                weak_tags = {}
                for s in failed_subs:
                    for tag in s['problem'].get('tags', []):
                        weak_tags[tag] = weak_tags.get(tag, 0) + 1
                        
                # Sort to find the Top 5 most failed topics
                sorted_weak_tags = sorted(weak_tags.items(), key=lambda item: item[1], reverse=True)[:5]
                
                if not sorted_weak_tags:
                    ai_roadmap = "<div class='alert alert-success text-center mt-4'><h4>🎉 Flawless!</h4><p>No failed submissions found in your recent history.</p></div>"
                else:
                    # 4. Format the data for the AI
                    tags_str = ", ".join([f"{tag} ({count} fails)" for tag, count in sorted_weak_tags])
                    top_tag = sorted_weak_tags[0][0] # The #1 most failed tag
                    
                    # 5. Generate the AI Roadmap
                    genai.configure(api_key=settings.GEMINI_API_KEY)
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    
                    prompt = f"""
                    You are an elite Competitive Programming Coach. 
                    Your student '{handle}' is analyzing their last 100 submissions. They have predominantly failed on problems with these tags: {tags_str}.
                    
                    Create a personalized, structured 4-week practice roadmap to fix these weaknesses.
                    
                    Format strictly in Markdown using this exact structure:
                    
                    ### 🎯 Diagnostic Summary
                    A 2-sentence explanation of *why* beginners typically struggle with {top_tag} and the other listed topics.
                    
                    ### 🗓️ 4-Week Action Plan
                    - **Week 1 (Foundations):** [Specific topics to review based on their worst tags]
                    - **Week 2 (Application):** [Specific problem types to practice]
                    - **Week 3 (Advanced):** [Combining concepts or optimization]
                    - **Week 4 (Mock Contests):** [Execution and timing strategy]
                    
                    ### 💡 Coach's Advice for {top_tag}
                    Give 2 highly technical, actionable coding tips specifically to prevent Time Limit Exceeded or Wrong Answers in {top_tag}.
                    
                    CRITICAL: Be encouraging but technical. Do not write filler intros/outros.
                    """
                    
                    ai_resp = model.generate_content(prompt)
                    ai_roadmap = markdown.markdown(ai_resp.text)
                    
            else:
                error_message = "Failed to fetch submission history from Codeforces."
                
        except Exception as e:
            if "429" in str(e).lower() or "quota" in str(e).lower():
                error_message = "AI is currently cooling down from rate limits. Please wait 60 seconds."
            else:
                error_message = f"System Error: {str(e)}"
                
    return render(request, 'core/weak_spot.html', {
        'ai_roadmap': ai_roadmap,
        'error_message': error_message,
        'handle': handle
    })