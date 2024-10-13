import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, UnidentifiedImageError
import numpy as np
from charm.toolbox.pairinggroup import PairingGroup, ZR, G1, G2, GT, pair
from charm.toolbox.secretutil import SecretUtil
from charm.toolbox.ABEnc import ABEnc, Input, Output

# Define types for CP-ABE
pk_t = {'g': G1, 'g2': G2, 'h': G1, 'f': G1, 'e_gg_alpha': GT}
mk_t = {'beta': ZR, 'g2_alpha': G2}
sk_t = {'D': G2, 'Dj': G2, 'Djp': G1, 'S': str}
ct_t = {'C_tilde': GT, 'C': G1, 'Cy': G1, 'Cyp': G2}

# Global variables
AttributeEntered = []
group = None
M_serialized = None

class CPabe_BSW07(ABEnc):
    def __init__(self, groupObj):
        ABEnc.__init__(self)
        global util, group
        util = SecretUtil(groupObj, verbose=False)
        group = groupObj

    @Output(pk_t, mk_t)
    def setup(self):
        g, gp = group.random(G1), group.random(G2)
        alpha, beta = group.random(ZR), group.random(ZR)
        g.initPP()
        gp.initPP()
        h = g ** beta
        f = g ** (~beta)
        e_gg_alpha = pair(g, gp ** alpha)
        pk = {'g': g, 'g2': gp, 'h': h, 'f': f, 'e_gg_alpha': e_gg_alpha}
        mk = {'beta': beta, 'g2_alpha': gp ** alpha}
        return (pk, mk)

    @Input(pk_t, mk_t, [str])
    @Output(sk_t)
    def keygen(self, pk, mk, S):
        r = group.random()
        g_r = pk['g2'] ** r
        D = (mk['g2_alpha'] * g_r) ** (1 / mk['beta'])
        D_j, D_j_pr = {}, {}
        for j in S:
            r_j = group.random()
            D_j[j] = g_r * (group.hash(j, G2) ** r_j)
            D_j_pr[j] = pk['g'] ** r_j
        return {'D': D, 'Dj': D_j, 'Djp': D_j_pr, 'S': S}

    @Input(pk_t, GT, str)
    @Output(ct_t)
    def encrypt(self, pk, M, policy_str):
        policy = util.createPolicy(policy_str)
        a_list = util.getAttributeList(policy)
        s = group.random(ZR)
        shares = util.calculateSharesDict(s, policy)

        C = pk['h'] ** s
        C_y, C_y_pr = {}, {}
        for i in shares.keys():
            j = util.strip_index(i)
            C_y[i] = pk['g'] ** shares[i]
            C_y_pr[i] = group.hash(j, G2) ** shares[i]

        return {
            'C_tilde': (pk['e_gg_alpha'] ** s) * M,
            'C': C,
            'Cy': C_y,
            'Cyp': C_y_pr,
            'policy': policy_str,
            'attributes': a_list
        }

    @Input(pk_t, sk_t, ct_t)
    @Output(GT)
    def decrypt(self, pk, sk, ct):
        policy = util.createPolicy(ct['policy'])
        pruned_list = util.prune(policy, sk['S'])
        if pruned_list == False:
            return False
        z = util.getCoefficients(policy)
        A = 1
        for i in pruned_list:
            j = i.getAttributeAndIndex()
            k = i.getAttribute()
            A *= (pair(ct['Cy'][j], sk['Dj'][k]) / pair(sk['Djp'][k], ct['Cyp'][j])) ** z[j]
        return ct['C_tilde'] / (pair(ct['C'], sk['D']) / A)

def blur_faces_and_hash(root, pk):
    global group
    # Ask the user to select an image file
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff"), ("All Files", "*.*")]
    )

    if not file_path:
        print("No file selected.")
        return None

    try:
        # Load the image using PIL
        image_pil = Image.open(file_path)
        # Convert PIL Image to OpenCV image format
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, AttributeError, ValueError, OSError) as e:
        messagebox.showerror("Error", f"Failed to open image:\n{file_path}\n\n{str(e)}")
        return None

    # Make copies of the original image for display and hashing
    original_image_cv = image_cv.copy()
    image_for_hashing_cv = image_cv.copy()

    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    # Convert image to grayscale
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Blur each face found
    for (x, y, w, h) in faces:
        # Extract the face region
        face_region = image_cv[y:y+h, x:x+w]
        # Apply Gaussian blur to the face region
        face_region_blurred = cv2.GaussianBlur(face_region, (99, 99), 30)
        # Replace the face region on the image
        image_cv[y:y+h, x:x+w] = face_region_blurred
        image_for_hashing_cv[y:y+h, x:x+w] = face_region_blurred

    # Convert images back to RGB format for PIL
    image_cv_rgb_blurred = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image_pil_blurred = Image.fromarray(image_cv_rgb_blurred)

    original_image_cv_rgb = cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2RGB)
    image_pil_original = Image.fromarray(original_image_cv_rgb)

    # Convert the blurred image for hashing
    image_for_hashing_rgb = cv2.cvtColor(image_for_hashing_cv, cv2.COLOR_BGR2RGB)
    image_pil_for_hashing = Image.fromarray(image_for_hashing_rgb)

    # Convert the blurred image to bytes
    from io import BytesIO
    image_bytes_io = BytesIO()
    image_pil_for_hashing.save(image_bytes_io, format='PNG')
    image_bytes = image_bytes_io.getvalue()

    # Hash the image bytes into ZR
    hashed_exponent = group.hash(image_bytes, ZR)
    print("Hashed exponent in ZR:")
    print(hashed_exponent)

    # Compute M in GT (pairing of g and g2 raised to hashed_exponent)
    M = pair(pk['g'], pk['g2']) ** hashed_exponent
    print("Message M in GT:")
    print(M)

    # Schedule the display of the image
    root.after(0, lambda: display_image_with_unblur(root, image_pil_blurred, image_pil_original))

    return M

def display_image_with_unblur(root, image_pil_blurred, image_pil_original):
    global cpabe, pk, mk, ct, my_set, M_serialized
    # Create a new Tkinter window
    window = tk.Toplevel(root)
    window.title("Image Viewer with Unblur Functionality")

    # Function to handle window close event
    def on_closing():
        root.destroy()

    window.protocol("WM_DELETE_WINDOW", on_closing)

    # Initialize variables to keep track of the current image
    current_state = {'is_blurred': True}

    # Resize images if they're too large for display
    max_size = 800
    img_width, img_height = image_pil_blurred.size
    resize_ratio = min(max_size / img_width, max_size / img_height, 1)
    display_width = int(img_width * resize_ratio)
    display_height = int(img_height * resize_ratio)
    image_pil_blurred_resized = image_pil_blurred.resize((display_width, display_height), Image.LANCZOS)
    image_pil_original_resized = image_pil_original.resize((display_width, display_height), Image.LANCZOS)

    # Convert images to ImageTk format
    photo_blurred = ImageTk.PhotoImage(image_pil_blurred_resized)
    photo_original = ImageTk.PhotoImage(image_pil_original_resized)

    # Create a Canvas to display the image
    canvas = tk.Canvas(window, width=display_width, height=display_height)
    canvas.pack()
    image_on_canvas = canvas.create_image(0, 0, anchor=tk.NW, image=photo_blurred)

    # Function to toggle between blurred and original images with attribute collection
    def toggle_image():
        global AttributeEntered, cpabe, pk, mk, ct, my_set, M_serialized
        if current_state['is_blurred']:
            # Collect attributes from the user
            attributes = collect_attributes(window)
            if attributes:
                AttributeEntered = attributes
                print("Collected attributes:")
                print(attributes)

                # Generate user's secret key based on attributes
                try:
                    sk = cpabe.keygen(pk, mk, AttributeEntered)
                    print("Secret Key:", sk)
                except Exception as e:
                    messagebox.showerror("Error", f"Key generation failed:\n{str(e)}")
                    return

                # Attempt to decrypt the ciphertext
                try:
                    rec_msg = cpabe.decrypt(pk, sk, ct)
                except Exception as e:
                    messagebox.showerror("Error", f"Decryption failed:\n{str(e)}")
                    return

                # Check if decryption was successful
                if rec_msg:
                    # Serialize the decrypted message for comparison
                    rec_msg_serialized = group.serialize(rec_msg)
                    if rec_msg_serialized == M_serialized:
                        # Decryption successful, unblur the image
                        canvas.itemconfig(image_on_canvas, image=photo_original)
                        toggle_button.config(text="Blur")
                        current_state['is_blurred'] = False
                        print("Image successfully unblurred.")
                    else:
                        # Decryption failed, show error message
                        messagebox.showerror("Error", "Incorrect attributes. Unable to unblur the image.")
                else:
                    # Decryption failed, show error message
                    messagebox.showerror("Error", "Decryption returned no result.")
            else:
                print("No attributes provided. Image remains blurred.")
        else:
            # Switch back to blurred image
            canvas.itemconfig(image_on_canvas, image=photo_blurred)
            toggle_button.config(text="Unblur")
            current_state['is_blurred'] = True

    # Create a button to toggle between images
    toggle_button = tk.Button(window, text="Unblur", command=toggle_image)
    toggle_button.pack(pady=10)

    # Keep references to the images to prevent garbage collection
    canvas.image_blurred = photo_blurred
    canvas.image_original = photo_original

def collect_attributes(parent_window):
    # Prompt the user to enter attributes in a simple dialog
    attrs_input = simpledialog.askstring("Input", "Enter attributes separated by commas:", parent=parent_window)
    if attrs_input:
        # Split the input string by commas and strip whitespace
        attrs = [attr.strip().upper() for attr in attrs_input.split(',') if attr.strip()]
        return attrs
    else:
        return None

if __name__ == "__main__":
    # Create the main Tkinter root window
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Initialize the pairing group and ABE scheme
    group = PairingGroup('SS512')
    cpabe = CPabe_BSW07(group)
    access_policy = '((aditya or amit) and (amit or suman))'
    (pk, mk) = cpabe.setup()

    # Call the function to blur faces and hash the image, passing 'pk'
    M = blur_faces_and_hash(root, pk)

    if M is None:
        print("Image processing failed.")
        exit()

    # Serialize M for comparison
    M_serialized = group.serialize(M)

    # Encrypt the message M under the access policy
    try:
        ct = cpabe.encrypt(pk, M, access_policy)
    except Exception as e:
        messagebox.showerror("Error", f"Encryption failed:\n{str(e)}")
        exit()

    # Store M_serialized for comparison
    my_set = set()
    my_set.add(M_serialized)

    # Start the main Tkinter event loop
    root.mainloop()
