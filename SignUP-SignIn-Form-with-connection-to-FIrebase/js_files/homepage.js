import { initializeApp } from "https://www.gstatic.com/firebasejs/11.2.0/firebase-app.js";
import { getAuth, onAuthStateChanged, signOut } from "https://www.gstatic.com/firebasejs/11.2.0/firebase-auth.js";
import { getFirestore, getDoc, doc } from "https://www.gstatic.com/firebasejs/11.2.0/firebase-firestore.js";

// Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyBKzklTetliRIv9UunBkZ-JfqV5mQnf3PU",
  authDomain: "login-form-784b8.firebaseapp.com",
  projectId: "login-form-784b8",
  storageBucket: "login-form-784b8.firebasestorage.app",
  messagingSenderId: "444698526985",
  appId: "1:444698526985:web:c032a38d5cd43af9cbbfd8",
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth();
const db = getFirestore();

// Function to display user data
function displayUserData(userData, email) {
  document.getElementById("loggedUserFName").innerText = userData.firstName || "N/A";
  document.getElementById("loggedUserLName").innerText = userData.lastName || "N/A";
  document.getElementById("loggedUserEmail").innerText = email || "N/A"; // Display the Gmail ID
}

// Function to log out the user
function handleLogout() {
  localStorage.removeItem("loggedInUserId");
  signOut(auth)
    .then(() => {
      window.location.href = "index.html"; // Redirect to the login page
    })
    .catch((error) => {
      console.error("Error Signing out:", error);
      alert("An error occurred while signing out. Please try again.");
    });
}

// Event listener for logout
const logoutButton = document.getElementById("logout");
if (logoutButton) {
  logoutButton.addEventListener("click", handleLogout);
}

// Monitor authentication state
onAuthStateChanged(auth, (user) => {
  if (user) {
    const loggedInUserId = localStorage.getItem("loggedInUserId");

    if (!loggedInUserId || user.uid !== loggedInUserId) {
      console.warn("No valid user session found. Redirecting to login...");
      window.location.href = "index.html";
      return;
    }

    const docRef = doc(db, "users", loggedInUserId);

    getDoc(docRef)
      .then((docSnap) => {
        if (docSnap.exists()) {
          const userData = docSnap.data();
          displayUserData(userData, user.email); // Pass email to display
        } else {
          console.error("No document found for the given user ID.");
          alert("User data not found. Please log in again.");
          window.location.href = "index.html";
        }
      })
      .catch((error) => {
        console.error("Error fetching user data:", error);
        alert("Failed to retrieve user data. Please try again.");
      });
  } else {
    console.warn("User is not authenticated. Redirecting to login...");
    window.location.href = "index.html";
  }
});
