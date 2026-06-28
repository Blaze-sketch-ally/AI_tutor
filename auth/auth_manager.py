import os
import bcrypt  # type: ignore

from database.supabase_client import supabase


class AuthManager:

    def __init__(self):

        self.users_folder = os.path.join(
            "data",
            "users"
        )

        os.makedirs(
            self.users_folder,
            exist_ok=True
        )

    # =====================================
    # PASSWORD HASHING
    # =====================================

    def hash_password(
        self,
        password: str
    ) -> str:

        hashed = bcrypt.hashpw(
            password.encode("utf-8"),
            bcrypt.gensalt()
        )

        return hashed.decode("utf-8")

    # =====================================
    # VERIFY PASSWORD
    # =====================================

    def verify_password(
        self,
        password: str,
        password_hash: str
    ) -> bool:

        return bcrypt.checkpw(
            password.encode("utf-8"),
            password_hash.encode("utf-8")
        )

    # =====================================
    # REGISTER
    # =====================================

    def register_user(
        self,
        username: str,
        password: str
    ):

        username = username.strip().lower()

        existing = (
            supabase
            .table("users")
            .select("id")
            .eq("username", username)
            .execute()
        )

        if existing.data:

            return {
                "success": False,
                "message": "Username already exists."
            }

        password_hash = self.hash_password(
            password
        )

        (
            supabase
            .table("users")
            .insert(
                {
                    "username": username,
                    "password_hash": password_hash
                }
            )
            .execute()
        )

        user_folder = os.path.join(
            self.users_folder,
            username
        )

        os.makedirs(
            user_folder,
            exist_ok=True
        )

        return {
            "success": True,
            "message": "Registration successful."
        }

    # =====================================
    # LOGIN
    # =====================================

    def login_user(
        self,
        username: str,
        password: str
    ):

        username = username.strip().lower()

        result = (
            supabase
            .table("users")
            .select("*")
            .eq("username", username)
            .execute()
        )

        if not result.data:

            return {
                "success": False,
                "message": "User not found."
            }

        user = result.data[0]

        if not self.verify_password(
            password,
            user["password_hash"]
        ):

            return {
                "success": False,
                "message": "Invalid password."
            }

        return {
            "success": True,
            "message": "Login successful.",
            "username": username,
            "user_id": user["id"]
        }

    # =====================================
    # USER EXISTS
    # =====================================

    def user_exists(
        self,
        username: str
    ):

        username = username.strip().lower()

        result = (
            supabase
            .table("users")
            .select("id")
            .eq("username", username)
            .execute()
        )

        return len(result.data) > 0

    # =====================================
    # DELETE USER
    # =====================================

    def delete_user(
        self,
        username: str
    ):

        username = username.strip().lower()

        result = (
            supabase
            .table("users")
            .select("id")
            .eq("username", username)
            .execute()
        )

        if not result.data:

            return {
                "success": False,
                "message": "User not found."
            }

        (
            supabase
            .table("users")
            .delete()
            .eq("username", username)
            .execute()
        )

        return {
            "success": True,
            "message": "User deleted."
        }

    # =====================================
    # LIST USERS
    # =====================================

    def list_users(
        self
    ):

        result = (
            supabase
            .table("users")
            .select("username")
            .order("username")
            .execute()
        )

        return [
            row["username"]
            for row in result.data
        ]